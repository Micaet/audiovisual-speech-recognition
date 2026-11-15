"""
Przetwarza LRW na pliki .npz
w tym samym formacie co MODALITY i Oulu.

Dla każdego klipu:
  - wideo 29 klatek 256x256, 25 kl./s (OpenCV) -> wycinek ust 96x96 (common_functions),
  - dźwięk PyAV -> mono 16 kHz, filtr górnoprzepustowy 80 Hz, int16,
  - granice słowa z pola "Duration" w .txt (słowo jest w środku klipu),
  - zapis lrw_npz\\ABOUT\\train\\ABOUT_00001.npz: video (29,96,96) uint8, audio (18560,) int16,
    usta_pkt (29,40,2), wykryto (29,), slowo, mowca ("nieznany"), zbior, zbior_danych ("LRW"),
    poczatek_slowa / koniec_slowa (w klatkach), meta (json).

Wymagania: numpy, scipy, opencv-python, "mediapipe==0.10.14", av==12.3.0 (PyAV).
"""
import argparse
import os
import sys
from pathlib import Path

import av
import cv2
import numpy as np
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_functions as cf 

CZESCI = {"train": "trening", "val": "walidacja", "test": "test"}


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    frs = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frs.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frs


def read_audio(path):
    """int16 16 kHz mono; filtr = ffmpeg "highpass=f=80" (dwubiegunowy, Q=0.707), jak w MODALITY i Oulu."""
    with av.open(str(path)) as cont:
        res = av.AudioResampler(format="flt", layout="mono", rate=cf.SR)
        kaw = []
        for fr in cont.decode(cont.streams.audio[0]):
            kaw += [f.to_ndarray().reshape(-1) for f in res.resample(fr)]
        kaw += [f.to_ndarray().reshape(-1) for f in res.resample(None)]
    w0 = 2 * np.pi * 80.0 / cf.SR
    al, c = np.sin(w0) / (2 * 0.707), np.cos(w0)
    b = np.array([(1 + c) / 2, -(1 + c), (1 + c) / 2])
    a = np.array([1 + al, -2 * c, 1 - al])
    x = lfilter(b / a[0], a / a[0], np.concatenate(kaw).astype(np.float64))
    return np.clip(np.round(x * 32768), -32768, 32767).astype(np.int16)


def read_duration(txt):
    for line in Path(txt).read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("Duration:"):
            return float(line.split()[1])
    return None


def process_clip(job):
    mp4, out, slowo, czesc = job
    frs = read_video(mp4)
    T = len(frs)
    if T < 5:
        return f"tylko {T} klatek"
    wynik = cf.usta(frs, cf.landmarks(frs))
    if wynik is None:
        return "nie wykryto twarzy w żadnej klatce"
    video, lips_norm, wykryto, side, center = wynik
    dur = read_duration(Path(mp4).with_suffix(".txt"))
    pol = (dur or 0) * cf.FPS / 2                       # słowo jest w środku klipu
    meta = dict(plik=Path(mp4).name, czas_slowa_s=dur, klatki=T, wykrycie=round(float(wykryto.mean()), 3),
                bok_wycinka_px=side, srodek_ust_px=[round(float(c), 1) for c in center.mean(0)],
                audio="int16, podzielić przez 32768")
    cf.zapisz(out, meta, video=video, audio=cf.dopasuj(read_audio(mp4), T * cf.SPF, np.int16),
              usta_pkt=lips_norm, wykryto=wykryto, slowo=slowo, mowca="nieznany", zbior=CZESCI[czesc],
              zbior_danych="LRW", poczatek_slowa=round(max(0.0, T / 2 - pol), 2),
              koniec_slowa=round(min(float(T), T / 2 + pol), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lrw", default=str(Path("lrw") / "lipread_mp4"))
    ap.add_argument("--out", default="lrw_npz")
    ap.add_argument("--procesy", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pomin = cf.nieudane(out)
    jobs = []
    for czesc in CZESCI:
        for w in sorted(e.name for e in os.scandir(args.lrw) if e.is_dir()):
            d = Path(args.lrw) / w / czesc
            for f in sorted(e.name for e in os.scandir(d) if e.name.endswith(".mp4")):
                dest = out / w / czesc / (f[:-4] + ".npz")
                if not dest.exists() and str(d / f) not in pomin:
                    jobs.append((str(d / f), str(dest), w, czesc))
    cf.uruchom(process_clip, jobs, out, args.procesy)


if __name__ == "__main__":
    main()
