"""
Przetwarza OuluVS na pliki .npz.

Dla każdego klipu:
  - wideo: ffmpeg usuwa przeplot (yadif) i poprawia proporcje pikseli PAL (720 -> 768 px), 25 kl./s,
    potem wycinek ust 96x96 (common_functions),
  - dźwięk: ffmpeg -> mono 16 kHz, filtr górnoprzepustowy 80 Hz, int16,
  - klip obejmuje całą frazę: poczatek_slowa = 0, koniec_slowa = liczba klatek,
  - zapis oulu_npz\\Excuse\\P004_Excuse1.npz: video (T,96,96) uint8, audio (T*640,) int16, usta_pkt (T,40,2),
    wykryto (T,), slowo ("Excuse me"), mowca ("P004"), zbior ("brak podziału"), zbior_danych ("OuluVS"),
    poczatek_slowa / koniec_slowa, label (0-9), powtorzenie, meta (json).

Wymagania: ffmpeg w PATH, numpy, opencv-python, "mediapipe==0.10.14".
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_functions as cf  

W, H = 768, 576                     
FRAZY = { 
    "Excuse": (0, "Excuse me"), "Goodby": (1, "Goodbye"), "Hello": (2, "Hello"),
    "How": (3, "How are you"), "Nice": (4, "Nice to meet you"), "Seeyou": (5, "See you"),
    "Sorry": (6, "I am sorry"), "Thank": (7, "Thank you"), "Time": (8, "Have a good time"),
    "Welcome": (9, "You are welcome"),
}
RE_PLIK = re.compile(r"^([A-Za-z]+?)(\d+)\.avi$", re.IGNORECASE)


def process_clip(job):
    avi, out, osoba, fraza, nr = job
    raw = cf.ffmpeg(["-i", avi, "-vf", f"yadif=0:-1:0,scale={W}:{H}", "-r", str(cf.FPS),
                     "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
    frs = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
    audio = np.frombuffer(cf.ffmpeg(["-i", avi, "-vn", "-af", "highpass=f=80", "-ac", "1",
                                     "-ar", str(cf.SR), "-f", "s16le", "-"]), np.int16)
    T = len(frs)
    if T < 5:
        return f"tylko {T} klatek"
    wynik = cf.usta(frs, cf.landmarks(frs))
    if wynik is None:
        return "nie wykryto twarzy w żadnej klatce"
    video, lips_norm, wykryto, side, center = wynik
    label, tekst = FRAZY[fraza]
    meta = dict(plik=f"{osoba}/{Path(avi).name}", klatki=T, dl_audio_s=round(len(audio) / cf.SR, 3),
                roznica_av_ms=round(1000 * (len(audio) / cf.SR - T / cf.FPS)),
                wykrycie=round(float(wykryto.mean()), 3), bok_wycinka_px=side,
                srodek_ust_px=[round(float(c), 1) for c in center.mean(0)],
                audio="int16, podzielić przez 32768")
    cf.zapisz(out, meta, video=video, audio=cf.dopasuj(audio, T * cf.SPF, np.int16), usta_pkt=lips_norm,
              wykryto=wykryto, slowo=tekst, mowca=osoba, zbior="brak podziału", zbior_danych="OuluVS",
              poczatek_slowa=0.0, koniec_slowa=float(T), label=label, powtorzenie=int(nr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--avi", default=str(Path("oulu") / "avi"), help="folder z wynikiem wypakuj_oulu.py")
    ap.add_argument("--out", default="oulu_npz")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pomin = cf.nieudane(out)
    jobs = []
    for osoba in sorted(e.name for e in os.scandir(args.avi) if e.is_dir()):
        for f in sorted(os.listdir(Path(args.avi) / osoba)):
            m = RE_PLIK.match(f)
            fraza = next((k for k in FRAZY if m and k.lower() == m.group(1).lower()), None)
            if fraza is None:
                continue
            src = Path(args.avi) / osoba / f
            dest = out / fraza / f"{osoba}_{fraza}{m.group(2)}.npz"
            if not dest.exists() and str(src) not in pomin:
                jobs.append((str(src), str(dest), osoba, fraza, m.group(2)))
    cf.uruchom(process_clip, jobs, out)


if __name__ == "__main__":
    main()
