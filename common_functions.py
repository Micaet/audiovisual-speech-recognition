"""
Funkcje wspólne dla przetworz_lrw.py, przetworz_modality.py i przetworz_oulu.py.

"""
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SR, FPS = 16000, 25
SPF = SR // FPS               # 640 próbek dźwięku na klatkę
ROI = 96                      # bok wycinka ust w pikselach
SKALA = 0.6                   # bok wycinka / szerokość twarzy
LIPS = None                   # detektor twarzy, tworzony przez init_lips() w każdym procesie


class Lips:
    """MediaPipe Face Mesh: 468 punktów twarzy w każdej klatce ."""

    def __init__(self):
        import mediapipe as mpipe
        self.idx = sorted({i for e in mpipe.solutions.face_mesh.FACEMESH_LIPS for i in e})   
        self.fm = mpipe.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                     refine_landmarks=False)

    def reset(self, shape):
        self.fm.process(np.zeros(shape, np.uint8))

    def __call__(self, frame):
        """frame: obraz RGB (H, W, 3). Zwraca (468, 2) w pikselach albo None, gdy nie ma twarzy."""
        h, w = frame.shape[:2]
        res = self.fm.process(frame)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        return np.array([(p.x * w, p.y * h) for p in lm[:468]], np.float32)


def init_lips():
    global LIPS
    import os
    import warnings
    warnings.filterwarnings("ignore")                 
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    cv2.setNumThreads(1)
    LIPS = Lips()


def ffmpeg(args, wejscie=None):
    return subprocess.run(["ffmpeg", "-v", "error", *args], input=wejscie,
                          capture_output=True, check=True).stdout


def interp_nan(x):
    """Uzupełnia klatki bez wykrytej twarzy (NaN) liniowo z sąsiednich klatek. None = brak twarzy wszędzie."""
    x = x.copy()
    flat = x.reshape(len(x), -1)
    t = np.arange(len(x))
    ok = ~np.isnan(flat[:, 0])
    if ok.sum() == 0:
        return None
    for j in range(flat.shape[1]):
        flat[:, j] = np.interp(t, t[ok], flat[ok, j])
    return flat.reshape(x.shape)


def smooth(x, k=5):
    """Średnia krocząca po czasie (5 klatek), żeby wycinek nie drgał."""
    pad = k // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    return np.array([xp[i:i + k].mean(0) for i in range(len(x))])


def crop_mouth(frames_rgb, pts, lip_idx, scale):
    """Wycinek ust 96x96 w skali szarości z każdej klatki.
    Środek = wygładzony środek punktów ust, bok = scale × szerokość twarzy (stały w obrębie klipu).
    Zwraca: wycinki (T,96,96) uint8, punkty ust względem środka w jednostkach szerokości twarzy (T,40,2),
            bok wycinka w pikselach oryginału, środki wycinka (T,2)."""
    lips = pts[:, lip_idx]
    center = smooth(lips.mean(1))
    face_w = np.linalg.norm(pts[:, 234] - pts[:, 454], axis=1)
    side = max(16, int(round(np.median(face_w) * scale)))
    half = side // 2
    out = np.empty((len(frames_rgb), ROI, ROI), np.uint8)
    for t, (fr, (cx, cy)) in enumerate(zip(frames_rgb, center)):
        g = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
        h, w = g.shape
        g = np.pad(g, half, mode="edge")
        x0, y0 = int(round(min(max(cx, 0), w - 1))), int(round(min(max(cy, 0), h - 1)))
        patch = g[y0:y0 + side, x0:x0 + side]
        out[t] = cv2.resize(patch, (ROI, ROI), interpolation=cv2.INTER_AREA)
    lips_norm = (lips - center[:, None]) / np.median(face_w)
    return out, lips_norm.astype(np.float32), side, center


def landmarks(frames):
    """Punkty twarzy dla całego klipu: (T,468,2), NaN tam, gdzie twarzy nie znaleziono."""
    LIPS.reset(frames[0].shape)
    pts = np.full((len(frames), 468, 2), np.nan, np.float32)
    for k, fr in enumerate(frames):
        p = LIPS(fr)
        if p is not None:
            pts[k] = p
    return pts


def usta(frames, pts):
    """Wycinki ust z klatek i ich punktów twarzy. None, gdy w żadnej klatce nie ma twarzy.
    Zwraca: video (T,96,96), usta_pkt (T,40,2), wykryto (T,), bok wycinka, środki wycinka."""
    wykryto = ~np.isnan(pts[:, 0, 0])
    pts_i = interp_nan(pts)
    if pts_i is None:
        return None
    video, lips_norm, side, center = crop_mouth(frames, pts_i, LIPS.idx, SKALA)
    return video, lips_norm, wykryto, side, center


def dopasuj(audio, n, dtype):
    """Dźwięk przycięty albo dopełniony zerami do n próbek."""
    a = np.zeros(n, dtype)
    a[:min(n, len(audio))] = audio[:n]
    return a


def zapisz(path, meta, **dane):
    """Zapis .npz przez plik tymczasowy: przerwanie w trakcie nie zostawia uszkodzonego pliku."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + "_tmp.npz")
    np.savez_compressed(tmp, **dane, meta=json.dumps(meta, ensure_ascii=False))
    tmp.replace(path)

def nieudane(out):
    """Klipy z nieudane.txt - przy kolejnych uruchomieniach są pomijane."""
    f = Path(out) / "nieudane.txt"
    return {l.split("\t")[0] for l in open(f, encoding="utf-8")} if f.exists() else set()


def _bezpiecznie(fn_job):
    """Błąd w jednym klipie nie zatrzymuje całości: zwraca (plik, opis problemu albo None)."""
    fn, job = fn_job
    try:
        return job[0], fn(job)
    except Exception as e:
        return job[0], f"{type(e).__name__}: {str(e)[:200]}"


def uruchom(fn, jobs, out, procesy=1):
    print(f"Do zrobienia: {len(jobs)} klipów, procesów: {procesy}", flush=True)
    if not jobs:
        return
    pool = mp.get_context("spawn").Pool(procesy, initializer=init_lips) if procesy > 1 else None
    if pool is None:
        init_lips()
    zadania = [(fn, j) for j in jobs]
    wyniki = pool.imap_unordered(_bezpiecznie, zadania, chunksize=4) if pool else map(_bezpiecznie, zadania)
    t0, bledy, co_ile = time.time(), 0, max(20, len(jobs) // 200)
    keep_awake(True)
    try:
        for i, (plik, problem) in enumerate(wyniki, 1):
            if problem:
                bledy += 1
                with open(Path(out) / "nieudane.txt", "a", encoding="utf-8") as fh:
                    fh.write(f"{plik}\t{problem}\n")
            if i % co_ile == 0 or i == len(jobs):
                zostalo = (time.time() - t0) / i * (len(jobs) - i) / 60
                print(f"  [{i}/{len(jobs)}] zostało ~{zostalo:.0f} min, błędy: {bledy}", flush=True)
    except KeyboardInterrupt:
        print("\nPrzerwano (Ctrl+C) - uruchom ponownie, żeby dokończyć.")
    finally:
        if pool:
            pool.terminate()
            pool.join()
        keep_awake(False)
    print(f"Koniec po {(time.time() - t0) / 60:.0f} min, błędy: {bledy}")


def keep_awake(on):
    """Windows: nie usypiaj komputera, dopóki skrypt działa."""
    if sys.platform != "win32":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
