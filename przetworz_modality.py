"""
Przetwarza ZIP-y MODALITY (sesje C) na pliki .npz: jeden plik na słowo.

Dla każdego ZIP-a (np. SPEAKER14_C1.zip):
  - granice słów z pliku .lab (czas w jednostkach 100 ns) + margines 0,6 s z każdej strony,
  - dźwięk prosto z ZIP-a: mikrofon AUD1 (50 cm od mówcy) i mikrofon laptopa, filtr 80 Hz, mono 16 kHz,
  - wideo kamery STRL prosto z ZIP-a (bez rozpakowywania 3,5 GB), dekodowane OD POCZĄTKU, bo w tych
    plikach nie da się przewijać; 100 -> 25 kl./s, 1080x1920 -> 540x960, wycinek ust 96x96 (common_functions),
  - zapis modality_npz\\SPEAKER14\\SPEAKER14_C1_017_OPEN.npz: video (T,96,96) uint8, audio (T*640,) float32,
    audio_lapt (T*640,) float32, usta_pkt (T,40,2), wykryto (T,), slowo, mowca, sesja, warunki, zbior,
    poczatek_slowa / koniec_slowa (w klatkach od początku klipu), meta (json),
  - plik SPEAKER14_C1.done = sesja gotowa, przy ponownym uruchomieniu pomijana.

Użycie (z folderu Dane, w środowisku avsr):
    python przetworz_modality.py --dane modality --out modality_npz
Wymagania: ffmpeg w PATH, numpy, opencv-python, "mediapipe==0.10.14".
"""
import argparse
import re
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_functions as cf   # noqa: E402

W, H = 540, 960                 # połowa rozdzielczości pionowego wideo 1080x1920
KAMERA, MIKROFON = "STRL", "AUD1"
PAD = 0.6                       # margines [s]: wokół środka każdego słowa mieści się okno 29 klatek jak w LRW


def subfile_url(zpath, name):
    """Adres dla ffmpeg wskazujący plik wewnątrz ZIP-a (pliki wideo są w ZIP-ie bez kompresji)."""
    info = zipfile.ZipFile(zpath).getinfo(name)
    if info.compress_type != 0:
        raise RuntimeError(f"{name} jest skompresowany w ZIP-ie – trzeba go rozpakować")
    with open(zpath, "rb") as f:
        f.seek(info.header_offset)
        h = f.read(30)
    n, e = struct.unpack("<HH", h[26:30])
    off = info.header_offset + 30 + n + e
    return f"subfile,,start,{off},end,{off + info.file_size},,:file:{Path(zpath).resolve()}"


def read_lab(zf, name):
    """[(początek_s, koniec_s, słowo), ...]; utf-8-sig, bo pliki .lab zaczynają się znacznikiem BOM."""
    words = []
    for line in zf.read(name).decode("utf-8-sig", errors="replace").splitlines():
        p = line.split()
        if len(p) >= 3 and p[0].isdigit() and p[1].isdigit():
            words.append((int(p[0]) / 1e7, int(p[1]) / 1e7, " ".join(p[2:])))
    return words


def read_audio(zf, name):
    raw = cf.ffmpeg(["-i", "pipe:0", "-af", "highpass=f=80", "-ac", "1", "-ar", str(cf.SR), "-f", "s16le",
                     "pipe:1"], wejscie=zf.read(name))
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def frames(url, stop_s, errlog):
    """Kolejne klatki 25 kl./s od początku nagrania: (numer, obraz RGB HxWx3)."""
    cmd = ["ffmpeg", "-v", "error", "-threads", "0", "-i", url, "-t", f"{stop_s:.2f}",
           "-vf", f"fps={cf.FPS},scale={W}:{H}", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    with open(errlog, "ab") as err:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        n, k = W * H * 3, 0
        try:
            while len(buf := p.stdout.read(n)) == n:
                yield k, np.frombuffer(buf, np.uint8).reshape(H, W, 3)
                k += 1
        finally:
            p.kill()
            p.wait()


def process_zip(zpath, outroot, errlog):
    zf = zipfile.ZipFile(zpath)
    names = zf.namelist()
    labs = [n for n in names if n.lower().endswith(".lab")]
    m = re.match(r"(SPEAKER\d+)_(C\d)", Path(zpath).stem, re.I)
    if not m:
        return {"status": "pominięto (to nie sesja C)"}
    mowca, sesja = m.group(1).upper(), m.group(2).upper()
    sid = f"{mowca}_{sesja}"
    info = dict(mowca=mowca, sesja=sesja, warunki="cisza" if int(sesja[1]) <= 3 else "halas",
                zbior="test" if Path(zpath).stem.lower().endswith("_test") else "trening")
    if not labs:
        return {"sesja": sid, "status": "brak etykiet .lab – pominięto"}

    outdir = outroot / mowca
    outdir.mkdir(parents=True, exist_ok=True)
    pick = lambda suffix: next(n for n in names if n.upper().endswith(suffix))
    # okna słów w klatkach 25 kl./s (słowo + margines PAD z każdej strony)
    wins = [dict(i=i, s=s, e=e, w=w, k0=max(0, int(np.floor((s - PAD) * cf.FPS))), k1=int(np.ceil((e + PAD) * cf.FPS)),
                 path=outdir / f"{sid}_{i:03d}_{re.sub(r'[^A-Za-z0-9]+', '_', w).strip('_') or 'X'}.npz")
            for i, (s, e, w) in enumerate(read_lab(zf, labs[0]))]
    todo = [x for x in wins if not x["path"].exists()]
    rates = []
    if todo:
        aud = read_audio(zf, pick(f"_{MIKROFON}.WAV"))
        lapt = read_audio(zf, pick("_LAPT.WAV"))
        rates = run_session(subfile_url(zpath, pick(f"_{KAMERA}.MKV")), todo, aud, lapt, errlog, info)

    saved = sum(x["path"].exists() for x in wins)
    if saved == len(wins):
        (outdir / f"{sid}.done").write_text(f"{saved} słów\n", encoding="utf-8")
    return {"sesja": sid, "slow_w_lab": len(wins), "zapisano": saved,
            "wykrycie_sr": round(float(np.mean(rates)), 3) if rates else "",
            "status": "GOTOWE" if saved == len(wins) else "częściowo"}


def run_session(url, todo, aud, lapt, errlog, info):
    """Dekoduje wideo od początku i zapisuje słowa, gdy tylko ich okno się skończy.
    W pamięci trzyma tylko klatki z okien, które jeszcze nie zostały zapisane."""
    buf = {}                       # numer klatki -> (obraz, punkty twarzy albo None)
    pending, rates = list(todo), []
    stop_k = max(x["k1"] for x in todo)

    def zamknij(x):
        r = save_word(x, [j for j in range(x["k0"], x["k1"] + 1) if j in buf], buf, aud, lapt, info)
        if r is not None:
            rates.append(r)

    for k, fr in frames(url, stop_k / cf.FPS + 0.5, errlog):
        if any(x["k0"] <= k <= x["k1"] for x in pending):
            buf[k] = (fr, cf.LIPS(fr))
        while pending and k >= pending[0]["k1"]:
            zamknij(pending.pop(0))
            first_needed = pending[0]["k0"] if pending else k + 1
            for j in [j for j in buf if j < first_needed]:
                del buf[j]
        if k >= stop_k:
            break
    for x in pending:                  # nagranie skończyło się wcześniej niż okno słowa
        zamknij(x)
    return rates


def save_word(x, ks, buf, aud, lapt, info):
    if len(ks) < 3:
        return None
    pts = np.full((len(ks), 468, 2), np.nan, np.float32)
    for t, j in enumerate(ks):
        if buf[j][1] is not None:
            pts[t] = buf[j][1]
    wynik = cf.usta([buf[j][0] for j in ks], pts)
    if wynik is None:
        return None
    video, lips_norm, wykryto, side, _ = wynik
    a0, n = ks[0] * cf.SPF, len(ks) * cf.SPF
    meta = dict(slowo=x["w"], nr=x["i"], kamera=KAMERA, mikrofon=MIKROFON,
                start_s=round(x["s"], 3), koniec_s=round(x["e"], 3), klatki=len(ks),
                wykrycie=round(float(wykryto.mean()), 3), bok_wycinka_px=side)
    cf.zapisz(x["path"], meta, video=video, audio=cf.dopasuj(aud[a0:a0 + n], n, np.float32),
              audio_lapt=cf.dopasuj(lapt[a0:a0 + n], n, np.float32), usta_pkt=lips_norm, wykryto=wykryto,
              slowo=x["w"], mowca=info["mowca"], sesja=info["sesja"], warunki=info["warunki"],
              zbior=info["zbior"], poczatek_slowa=round(x["s"] * cf.FPS - ks[0], 2),
              koniec_slowa=round(x["e"] * cf.FPS - ks[0], 2))
    return float(wykryto.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dane", default="modality", help="folder z ZIP-ami")
    ap.add_argument("--out", default="modality_npz")
    args = ap.parse_args()

    outroot = Path(args.out)
    outroot.mkdir(parents=True, exist_ok=True)
    zips = sorted(Path(args.dane).glob("*.zip"))
    print(f"{len(zips)} ZIP-ów w {Path(args.dane).resolve()}")
    cf.init_lips()
    for zp in zips:
        m = re.match(r"(SPEAKER\d+)_(C\d)", zp.stem, re.I)
        if m and (outroot / m.group(1).upper() / f"{m.group(1).upper()}_{m.group(2).upper()}.done").exists():
            print(f"{zp.name}: już przetworzony (.done)")
            continue
        print(f"{zp.name}: przetwarzam...", flush=True)
        try:
            r = process_zip(zp, outroot, outroot / "ffmpeg_bledy.log")
        except Exception as e:          # jeden zepsuty ZIP nie zatrzymuje reszty
            r = {"sesja": zp.stem, "status": f"BŁĄD: {type(e).__name__}: {e}"}
        print("   ", r, flush=True)


if __name__ == "__main__":
    main()
