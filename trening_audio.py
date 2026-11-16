"""
Trening modelu audio 1 (spektrogram melowy + mała sieć CNN) i test w szumie.

Ustawienia są w bloku USTAWIENIA poniżej. Wyniki trafiają do wyniki\\<NAZWA>\\:
  - historia.csv          - po każdej epoce: strata, dokładność top-1/top-5 (trening i walidacja), lr, czas, pamięć
  - wykres_nauki.png      - odświeżany po każdej epoce (strata, dokładność, top-5, lr)
  - model_najlepszy.pt    - wagi z najlepszą dokładnością walidacji (czyste nagrania)
  - model_ostatni.pt      - pełny stan treningu (do wznowienia po przerwaniu: WZNOW = True)
  - po treningu test najlepszego modelu na części test:
      czyste nagrania: podsumowanie.txt / .json, wyniki_slow.csv (czułość, precyzja, F1 dla każdego słowa),
          najczestsze_pomylki.csv, macierz_pomylek.png, czulosc_slow.png, pewnosc.png
      w szumie: wyniki_szum.csv (dokładność, top-5, WER, strata dla każdego rodzaju szumu i SNR), wykres_snr.png
Szum: biały albo "babble" (suma kilku innych nagrań z tej samej części LRW), dodawany przez zaszumianie.py
tak, żeby SNR na fragmencie ze słowem był równy zadanemu. W teście szum zależy tylko od nagrania,
rodzaju i SNR, więc każdy model dostaje identycznie zaszumione nagrania.
Wymagania: torch z CUDA, numpy, matplotlib; zaszumianie.py w tym samym folderze.
"""
import csv
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
import numpy as np
import torch
import torch.nn as nn

TU = Path(__file__).resolve().parent
sys.path.insert(0, str(TU))
import zaszumianie as zs  # noqa: E402

# ======================= USTAWIENIA =======================
NAZWA = "audio_100"
EPOKI = 30
BATCH = 64
LR = 1e-3
WORKERS = 4
AUGMENTACJA = True
P_SZUM = 0.5
SNR_TRENING = (-5, 20)
SNR_TEST = [20, 10, 5, 0, -5]
TYPY_SZUMU = ["bialy", "babble"]
GLOSY_BABBLE = 6
ZIARNO = 1664
WZNOW = False
LRW = "lrw_npz"
LISTA_SLOW = "slowa_100.txt"
# ==========================================================

SR = 16000


def wczytaj(plik):
    """Dźwięk z pliku .npz jako float32 w zakresie -1..1 oraz granice słowa (w klatkach 25 kl./s)."""
    d = np.load(plik)
    return d["audio"].astype(np.float32) / 32768.0, (float(d["poczatek_slowa"]), float(d["koniec_slowa"]))


class LRWAudio(torch.utils.data.Dataset):
    """Nagrania jednej części LRW. augmentacja=True: z prawdopodobieństwem P_SZUM losowy szum i SNR.
    szum=(rodzaj, snr): każde nagranie zaszumione zawsze tak samo (do testu)."""

    def __init__(self, folder, slowa, czesc, augmentacja=False, szum=None):
        self.augmentacja, self.szum = augmentacja, szum
        self.pliki = [(str(p), i) for i, w in enumerate(slowa) for p in sorted((folder / w / czesc).glob("*.npz"))]
        rng = np.random.default_rng(ZIARNO)
        wybrane = rng.choice(len(self.pliki), min(1000, len(self.pliki)), replace=False)
        self.glosy = np.stack([wczytaj(self.pliki[k][0])[0] for k in wybrane])

    def __len__(self):
        return len(self.pliki)

    def __getitem__(self, i):
        plik, klasa = self.pliki[i]
        x, granice = wczytaj(plik)
        if self.szum is not None:
            rodzaj, snr = self.szum
            rng = np.random.default_rng(zs.ziarno(Path(plik).name, rodzaj, snr))
        elif self.augmentacja and np.random.rand() < P_SZUM:
            rng = np.random.default_rng(np.random.randint(2**31))
            rodzaj, snr = rng.choice(TYPY_SZUMU), rng.uniform(*SNR_TRENING)
        else:
            return torch.from_numpy(x), klasa
        x = zs.zaszum(x, rodzaj, snr, rng, granice, self.glosy, GLOSY_BABBLE)
        return torch.from_numpy(x), klasa


class LogMel(nn.Module):
    """Spektrogram melowy w skali logarytmicznej: 80 pasm, okno 25 ms (400 próbek), krok 10 ms (160 próbek)."""

    def __init__(self, n_mel=80, n_fft=512, okno=400, krok=160):
        super().__init__()
        self.n_fft, self.okno, self.krok = n_fft, okno, krok
        hz_na_mel = lambda f: 2595 * np.log10(1 + f / 700)
        mel_na_hz = lambda m: 700 * (10 ** (m / 2595) - 1)
        f = mel_na_hz(np.linspace(hz_na_mel(0), hz_na_mel(SR / 2), n_mel + 2))
        f_fft = np.linspace(0, SR / 2, n_fft // 2 + 1)
        filtry = np.zeros((n_mel, len(f_fft)), np.float32)
        for k in range(n_mel):
            rosnie = (f_fft - f[k]) / (f[k + 1] - f[k])
            maleje = (f[k + 2] - f_fft) / (f[k + 2] - f[k + 1])
            filtry[k] = np.clip(np.minimum(rosnie, maleje), 0, None)
        self.register_buffer("filtry", torch.from_numpy(filtry))
        self.register_buffer("hann", torch.hann_window(okno))

    def forward(self, x):                                        # (B, N)
        with torch.autocast(x.device.type, enabled=False):
            S = torch.stft(x.float(), self.n_fft, self.krok, self.okno, self.hann, return_complex=True)
            mel = torch.log(self.filtry @ S.abs() ** 2 + 1e-6)   # (B, 80, ramki)
            sr = mel.mean((1, 2), keepdim=True)
            od = mel.std((1, 2), keepdim=True)
        return (mel - sr) / (od + 1e-5)


def blok(we, wy):
    return nn.Sequential(
        nn.Conv2d(we, wy, 3, padding=1, bias=False), nn.BatchNorm2d(wy), nn.ReLU(),
        nn.Conv2d(wy, wy, 3, padding=1, bias=False), nn.BatchNorm2d(wy), nn.ReLU(),
        nn.MaxPool2d(2))


class ModelAudio(nn.Module):
    def __init__(self, n_klas):
        super().__init__()
        self.mel = LogMel()
        self.cnn = nn.Sequential(blok(1, 32), blok(32, 64), blok(64, 128), blok(128, 256))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, n_klas)

    def cechy(self, x):                         # (B, 18560)
        x = self.mel(x)[:, None]                # (B, 1, 80, 117)
        x = self.cnn(x)                         # (B, 256, 5, 7)
        return x.mean((2, 3))                   # (B, 256)

    def forward(self, x):
        return self.fc(self.dropout(self.cechy(x)))


def jedna_epoka(model, dl, dev, strata, opt=None, scaler=None):
    """Trening albo ocena. Zwraca stratę, top-1, top-5 oraz (dla oceny) przewidywania."""
    trening = opt is not None
    model.train(trening)
    suma, top1, top5, n = 0.0, 0, 0, 0
    pred, praw, pewn = [], [], []
    t0 = time.time()
    with torch.set_grad_enabled(trening):
        for i, (x, y) in enumerate(dl, 1):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                out = model(x)
                loss = strata(out, y)
            if trening:
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            suma += loss.item() * len(y)
            top = out.topk(5, 1).indices
            top1 += (top[:, 0] == y).sum().item()
            top5 += (top == y[:, None]).any(1).sum().item()
            n += len(y)
            if not trening:
                p = out.float().softmax(1)
                pred.append(top[:, 0].cpu())
                praw.append(y.cpu())
                pewn.append(p.max(1).values.cpu())
            elif i % 100 == 0:
                t = time.time() - t0
                print(f"    [{i}/{len(dl)}] strata {suma / n:.3f}, dokładność {top1 / n:.1%}, "
                      f"do końca epoki ~{t / i * (len(dl) - i) / 60:.1f} min", flush=True)
    wynik = dict(strata=suma / n, top1=top1 / n, top5=top5 / n)
    if not trening:
        wynik.update(pred=torch.cat(pred).numpy(), praw=torch.cat(praw).numpy(), pewn=torch.cat(pewn).numpy())
    return wynik


def wykres_nauki(hist, plik):
    ep = [h["epoka"] for h in hist]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(ep, [h["strata_tr"] for h in hist], "o-", label="trening")
    ax[0, 0].plot(ep, [h["strata_val"] for h in hist], "o-", label="walidacja")
    ax[0, 0].set(title="Strata (entropia krzyżowa)", xlabel="epoka")
    ax[0, 1].plot(ep, [100 * h["top1_tr"] for h in hist], "o-", label="trening")
    ax[0, 1].plot(ep, [100 * h["top1_val"] for h in hist], "o-", label="walidacja")
    ax[0, 1].set(title="Dokładność top-1 [%]", xlabel="epoka")
    ax[1, 0].plot(ep, [100 * h["top5_tr"] for h in hist], "o-", label="trening")
    ax[1, 0].plot(ep, [100 * h["top5_val"] for h in hist], "o-", label="walidacja")
    ax[1, 0].set(title="Dokładność top-5 [%]", xlabel="epoka")
    ax[1, 1].plot(ep, [h["lr"] for h in hist], "o-", color="gray")
    ax[1, 1].set(title="Współczynnik uczenia", xlabel="epoka")
    for a in ax.flat:
        a.grid(alpha=0.3)
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        if a.get_legend_handles_labels()[0]:
            a.legend()
    fig.tight_layout()
    fig.savefig(plik, dpi=120)
    plt.close(fig)


def raport_testu(w, slowa, out, info):
    """Statystyki i wykresy z testu najlepszego modelu na czystych nagraniach."""
    K = len(slowa)
    M = np.zeros((K, K), int)
    np.add.at(M, (w["praw"], w["pred"]), 1)
    tp = np.diag(M)
    czulosc = tp / np.maximum(M.sum(1), 1)
    precyzja = tp / np.maximum(M.sum(0), 1)
    f1 = 2 * precyzja * czulosc / np.maximum(precyzja + czulosc, 1e-9)
    with open(out / "wyniki_slow.csv", "w", newline="", encoding="utf-8") as f:
        c = csv.writer(f)
        c.writerow(["slowo", "nagran", "trafien", "czulosc", "precyzja", "f1"])
        for k in np.argsort(-czulosc):
            c.writerow([slowa[k], M[k].sum(), tp[k], f"{czulosc[k]:.4f}", f"{precyzja[k]:.4f}", f"{f1[k]:.4f}"])
    pomylki = sorted(((M[a, b], slowa[a], slowa[b]) for a in range(K) for b in range(K) if a != b and M[a, b]),
                     reverse=True)
    with open(out / "najczestsze_pomylki.csv", "w", newline="", encoding="utf-8") as f:
        c = csv.writer(f)
        c.writerow(["prawdziwe", "rozpoznane_jako", "ile_razy"])
        c.writerows([(a, b, n) for n, a, b in pomylki[:50]])

    fig, ax = plt.subplots(figsize=(12, 11))
    im = ax.imshow(M / np.maximum(M.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
    if K <= 40:
        ax.set_xticks(range(K), slowa, rotation=90, fontsize=7)
        ax.set_yticks(range(K), slowa, fontsize=7)
    ax.set(title="Macierz pomyłek (test, znormalizowana)", xlabel="rozpoznane słowo", ylabel="prawdziwe słowo")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "macierz_pomylek.png", dpi=150)
    plt.close(fig)

    kol = np.argsort(czulosc)
    n = min(15, K // 2)
    wyb = np.concatenate([kol[:n], kol[-n:]])
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(wyb) + 1.5))
    ax.barh(range(len(wyb)), 100 * czulosc[wyb], color=["#c0504d"] * n + ["#4f81bd"] * n)
    ax.set_yticks(range(len(wyb)), [slowa[k] for k in wyb], fontsize=8)
    ax.axvline(100 * czulosc.mean(), color="k", ls="--", lw=1, label=f"średnia czułość {czulosc.mean():.1%}")
    ax.set(title=f"{n} słów o najniższej i {n} o najwyższej czułości (test)", xlabel="czułość [%]", xlim=(0, 100))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "czulosc_slow.png", dpi=150)
    plt.close(fig)

    ok = w["pred"] == w["praw"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(w["pewn"][ok], bins=20, range=(0, 1), alpha=0.7, label="trafienia")
    ax.hist(w["pewn"][~ok], bins=20, range=(0, 1), alpha=0.7, label="pomyłki")
    ax.set(title="Pewność modelu (największe prawdopodobieństwo)", xlabel="pewność", ylabel="liczba nagrań")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "pewnosc.png", dpi=150)
    plt.close(fig)

    return dict(info, test_nagran=len(ok), test_strata=round(w["strata"], 4), test_top1=round(w["top1"], 4),
                test_top5=round(w["top5"], 4), test_WER=round(1 - w["top1"], 4),
                test_czulosc_makro=round(float(czulosc.mean()), 4),
                test_precyzja_makro=round(float(precyzja.mean()), 4), test_F1_makro=round(float(f1.mean()), 4),
                sr_pewnosc_trafien=round(float(w["pewn"][ok].mean()), 4) if ok.any() else None,
                sr_pewnosc_pomylek=round(float(w["pewn"][~ok].mean()), 4) if (~ok).any() else None,
                slowo_najwyzsza_czulosc=slowa[kol[-1]], slowo_najnizsza_czulosc=slowa[kol[0]],
                najczestsza_pomylka=f"{pomylki[0][1]} -> {pomylki[0][2]} ({pomylki[0][0]}x)" if pomylki else None)


def raport_szumu(wyniki, czysty, out):
    """Tabela i wykres dokładności w zależności od rodzaju szumu i SNR."""
    with open(out / "wyniki_szum.csv", "w", newline="", encoding="utf-8") as f:
        c = csv.writer(f)
        c.writerow(["szum", "snr_db", "dokladnosc", "top5", "WER", "strata"])
        c.writerow(["brak", "", f"{czysty['top1']:.4f}", f"{czysty['top5']:.4f}", f"{1 - czysty['top1']:.4f}",
                    f"{czysty['strata']:.4f}"])
        for (rodzaj, snr), w in wyniki.items():
            c.writerow([rodzaj, snr, f"{w['top1']:.4f}", f"{w['top5']:.4f}", f"{1 - w['top1']:.4f}",
                        f"{w['strata']:.4f}"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for rodzaj in TYPY_SZUMU:
        snr = sorted(s for r, s in wyniki if r == rodzaj)
        nazwa = "szum biały" if rodzaj == "bialy" else f"szum {rodzaj}"
        ax.plot(snr, [100 * wyniki[(rodzaj, s)]["top1"] for s in snr], "o-", label=nazwa)
    ax.axhline(100 * czysty["top1"], color="k", ls="--", lw=1, label="bez szumu")
    ax.set(title="Dokładność modelu audio w szumie (test)", xlabel="SNR [dB]", ylabel="dokładność [%]", ylim=(0, 102))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "wykres_snr.png", dpi=150)
    plt.close(fig)
    return {f"test_top1_{r}_{s}dB": round(w["top1"], 4) for (r, s), w in wyniki.items()}


def ziarno_procesu(_):
    np.random.seed(torch.initial_seed() % 2**32)


def main():
    random.seed(ZIARNO)
    np.random.seed(ZIARNO)
    torch.manual_seed(ZIARNO)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    karta = torch.cuda.get_device_name(0) if dev == "cuda" else "procesor"
    print("Urządzenie:", karta if dev == "cuda" else "procesor (brak CUDA - trening będzie bardzo wolny!)")
    lrw = TU / LRW
    slowa = ([l.strip() for l in open(lrw / LISTA_SLOW, encoding="utf-8") if l.strip()] if LISTA_SLOW
             else sorted(p.name for p in lrw.iterdir() if p.is_dir()))
    out = TU / "wyniki" / NAZWA
    if (out / "model_ostatni.pt").exists() and not WZNOW:
        raise SystemExit(f"{out} zawiera już trening. Zmień NAZWA albo ustaw WZNOW = True.")
    out.mkdir(parents=True, exist_ok=True)

    print("Wczytywanie list nagrań i głosów do szumu babble...")
    ds = {c: LRWAudio(lrw, slowa, c, augmentacja=AUGMENTACJA and c == "train") for c in ("train", "val", "test")}
    dl = {c: torch.utils.data.DataLoader(ds[c], batch_size=BATCH, shuffle=c == "train", num_workers=WORKERS,
                                         pin_memory=True, drop_last=c == "train", persistent_workers=WORKERS > 0,
                                         worker_init_fn=ziarno_procesu)
          for c in ("train", "val")}
    print(f"{len(slowa)} słów | nagrania: trening {len(ds['train'])}, walidacja {len(ds['val'])}, "
          f"test {len(ds['test'])}")

    model = ModelAudio(len(slowa)).to(dev)
    n_param = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOKI)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    strata = nn.CrossEntropyLoss()
    hist, start = [], 0
    if WZNOW and (out / "model_ostatni.pt").exists():
        st = torch.load(out / "model_ostatni.pt", map_location=dev, weights_only=False)
        for obj, k in ((model, "model"), (opt, "opt"), (sched, "sched"), (scaler, "scaler")):
            obj.load_state_dict(st[k])
        hist, start = st["hist"], st["epoka"] + 1
        print(f"Wznowiono po epoce {start}.")
    print(f"Parametry modelu: {n_param / 1e6:.2f} mln, epok: {EPOKI}, paczka: {BATCH}\n")

    for ep in range(start, EPOKI):
        t0 = time.time()
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        lr = opt.param_groups[0]["lr"]
        print(f"Epoka {ep + 1}/{EPOKI} (lr {lr:.2e})")
        tr = jedna_epoka(model, dl["train"], dev, strata, opt, scaler)
        t_tr = time.time() - t0
        val = jedna_epoka(model, dl["val"], dev, strata)
        sched.step()
        h = dict(epoka=ep + 1, lr=lr, strata_tr=tr["strata"], top1_tr=tr["top1"], top5_tr=tr["top5"],
                 strata_val=val["strata"], top1_val=val["top1"], top5_val=val["top5"],
                 czas_min=(time.time() - t0) / 60, klipy_na_s=len(ds["train"]) / t_tr,
                 pamiec_gb=torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0)
        najlepsza = not hist or h["top1_val"] > max(x["top1_val"] for x in hist)
        hist.append(h)
        if najlepsza:
            torch.save(model.state_dict(), out / "model_najlepszy.pt")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), hist=hist, epoka=ep), out / "model_ostatni.pt")
        with open(out / "historia.csv", "w", newline="", encoding="utf-8") as f:
            c = csv.DictWriter(f, fieldnames=list(h))
            c.writeheader()
            c.writerows(hist)
        wykres_nauki(hist, out / "wykres_nauki.png")
        zostalo = (EPOKI - ep - 1) * h["czas_min"] / 60
        print(f"  => strata {h['strata_tr']:.3f} / {h['strata_val']:.3f} | top-1 {h['top1_tr']:.1%} / "
              f"{h['top1_val']:.1%} | top-5 val {h['top5_val']:.1%} | {h['czas_min']:.1f} min"
              f"{' | NAJLEPSZY' if najlepsza else ''} | do końca ~{zostalo:.1f} h\n", flush=True)

    model.load_state_dict(torch.load(out / "model_najlepszy.pt", map_location=dev))

    def test(szum):
        ds["test"].szum = szum
        dl_test = torch.utils.data.DataLoader(ds["test"], batch_size=BATCH, num_workers=WORKERS, pin_memory=True)
        return jedna_epoka(model, dl_test, dev, strata)

    print("Test najlepszego modelu: czyste nagrania...")
    czysty = test(None)
    wyniki = {}
    for rodzaj in TYPY_SZUMU:
        for snr in SNR_TEST:
            wyniki[(rodzaj, snr)] = test((rodzaj, snr))
            print(f"  szum {rodzaj:6s} {snr:3d} dB: dokładność {wyniki[(rodzaj, snr)]['top1']:.1%}", flush=True)

    best = max(hist, key=lambda x: x["top1_val"])
    info = dict(nazwa=NAZWA, karta=karta, slow=len(slowa), nagran_trening=len(ds["train"]),
                nagran_walidacja=len(ds["val"]), parametry_mln=round(n_param / 1e6, 3), epok=len(hist),
                paczka=BATCH, lr=LR, augmentacja=AUGMENTACJA, p_szum=P_SZUM, snr_trening=list(SNR_TRENING),
                ziarno=ZIARNO, najlepsza_epoka=best["epoka"],
                walidacja_top1=round(best["top1_val"], 4), walidacja_top5=round(best["top5_val"], 4),
                trening_top1_ostatnia=round(hist[-1]["top1_tr"], 4),
                sr_czas_epoki_min=round(float(np.mean([x["czas_min"] for x in hist])), 2),
                laczny_czas_h=round(sum(x["czas_min"] for x in hist) / 60, 2),
                max_pamiec_gb=round(max(x["pamiec_gb"] for x in hist), 2))
    podsumowanie = raport_testu(czysty, slowa, out, info)
    podsumowanie.update(raport_szumu(wyniki, czysty, out))
    (out / "podsumowanie.json").write_text(json.dumps(podsumowanie, indent=2, ensure_ascii=False), encoding="utf-8")
    tekst = "\n".join(f"{k:28s} {v}" for k, v in podsumowanie.items())
    (out / "podsumowanie.txt").write_text(tekst + "\n", encoding="utf-8")
    print("\n" + tekst)
    print(f"\nWszystko zapisane w {out}")


if __name__ == "__main__":
    main()
