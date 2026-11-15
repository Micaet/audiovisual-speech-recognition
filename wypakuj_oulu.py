"""
Wypakowuje z zaszyfrowanego zipa tylko małe pliki AVI pojedynczych wypowiedzi
(wideo + dźwięk).

Użycie:
    python wypakuj_oulu.py --dane FOLDER_Z_ZIPEM --out oulu
Wynik: oulu\\avi\\P004\\Hello1.avi. Pliki, które już istnieją, są pomijane.
"""
import argparse
import getpass
import re
import sys
import zipfile
from pathlib import Path

ZIP_A = "A_16-12062006.zip"
RE_AVI = re.compile(r"^[^/]+/(P\d{3})/([A-Za-z]+)(\d+)/\2\3\.avi$", re.IGNORECASE)  


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dane", default=".", help="folder z ZIP-em (domyślnie bieżący)")
    ap.add_argument("--out", default=None, help="folder wynikowy (domyślnie <dane>/oulu)")
    args = ap.parse_args()

    out = (Path(args.out) if args.out else Path(args.dane) / "oulu") / "avi"
    pwd = getpass.getpass("Hasło do ZIP-ów OuluVS: ").encode()
    with zipfile.ZipFile(Path(args.dane) / ZIP_A) as zf:
        try:                            
            with zf.open(next(i for i in zf.infolist() if not i.is_dir()), pwd=pwd) as f:
                f.read(1024)
        except RuntimeError as e:
            if "password" in str(e).lower():
                sys.exit("Złe hasło.")
            raise
        items = [(i, m) for i in zf.infolist() if (m := RE_AVI.match(i.filename))
                 and m.group(2) not in ("Left", "Right")]
        print(f"{len(items)} plików AVI -> {out}")
        nowe = 0
        for k, (info, m) in enumerate(items, 1):
            dest = out / m.group(1).upper() / f"{m.group(2)}{m.group(3)}.avi"
            if dest.exists() and dest.stat().st_size == info.file_size:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".avi.part")
            with zf.open(info, pwd=pwd) as src, open(tmp, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            tmp.replace(dest)
            nowe += 1
            if k % 50 == 0:
                print(f"  {k}/{len(items)}", flush=True)
    print(f"Gotowe: wypakowano {nowe}, pominięto (już były) {len(items) - nowe}.")


if __name__ == "__main__":
    main()
