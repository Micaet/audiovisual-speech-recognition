"""
Trening modelu wideo 1 (splot 3D + ResNet-18 + BiGRU).

Ustawienia są w bloku USTAWIENIA poniżej. Wyniki trafiają do wyniki\\<NAZWA>\\:
  - historia.csv          - po każdej epoce: strata, dokładność top-1/top-5 (trening i walidacja), lr, czas, pamięć
  - wykres_nauki.png      - odświeżany po każdej epoce (strata, dokładność, top-5, lr)
  - model_najlepszy.pt    - wagi z najlepszą dokładnością walidacji
  - model_ostatni.pt      - pełny stan treningu (do wznowienia po przerwaniu: WZNOW = True)
  - po treningu test najlepszego modelu na części test:
      podsumowanie.txt / .json, wyniki_slow.csv (czułość, precyzja, F1 dla każdego słowa),
      najczestsze_pomylki.csv, macierz_pomylek.png, czulosc_slow.png, pewnosc.png
Wymagania: torch i torchvision z CUDA, numpy, matplotlib.
"""
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")        
import matplotlib.pyplot as plt
import matplotlib.ticker 
import numpy as np
import torch
import torch.nn as nn
import torchvision

# ======================= USTAWIENIA =======================
NAZWA = "wideo_100"              
EPOKI = 30
BATCH = 32
LR = 3e-4                        
WEIGHT_DECAY = 1e-2
WORKERS = 4                      
AUGMENTACJA = True               
WZNOW = False                    
LRW = "lrw_npz"                  
LISTA_SLOW = "slowa_100.txt"    
# ==========================================================

TU = Path(__file__).resolve().parent
SREDNIA, ODCHYLENIE = 0.42, 0.17          


class LRWDataset(torch.utils.data.Dataset):
    def __init__(self, folder, slowa, czesc, augmentacja=False):
        self.augmentacja = augmentacja
        self.pliki = [(str(p), i) for i, w in enumerate(slowa) for p in sorted((folder / w / czesc).glob("*.npz"))]

    def __len__(self):
        return len(self.pliki)

    def __getitem__(self, i):
        plik, klasa = self.pliki[i]
        x = np.load(plik)["video"].astype(np.float32) / 255.0          # (29, 96, 96)
        if self.augmentacja:
            dy, dx = np.random.randint(0, 9, 2)
            x = x[:, dy:dy + 88, dx:dx + 88]
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1]
        else:
            x = x[:, 4:92, 4:92]
        x = (x - SREDNIA) / ODCHYLENIE
        return torch.from_numpy(x.copy())[None], klasa                  # (1, 29, 88, 88)


class ModelWideo(nn.Module):
    def __init__(self, n_klas):
        super().__init__()
        self.front3d = nn.Sequential(
            nn.Conv3d(1, 64, (5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64), nn.ReLU(),
            nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)))
        resnet = torchvision.models.resnet18()
        self.resnet = nn.Sequential(*list(resnet.children())[4:-1]) 
        self.gru = nn.GRU(512, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, n_klas)

    def forward(self, x):                       # (B, 1, T, 88, 88)
        B, T = x.shape[0], x.shape[2]
        x = self.front3d(x)                     # (B, 64, T, 22, 22)
        x = x.transpose(1, 2).reshape(B * T, 64, x.shape[3], x.shape[4])
        x = self.resnet(x).reshape(B, T, 512)
        x, _ = self.gru(x)
        return self.fc(x.mean(1))


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
    """Statystyki i wykresy z testu najlepszego modelu."""
    K = len(slowa)
    M = np.zeros((K, K), int)                          # macierz pomyłek: wiersz = prawda, kolumna = odpowiedź
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

    # macierz pomyłek (znormalizowana wierszami)
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

    # słowa o najniższej i najwyższej czułości
    kol = np.argsort(czulosc)
    n = min(15, K // 2)
    wyb = np.concatenate([kol[:n], kol[-n:]])
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(wyb) + 1.5))
    ax.barh(range(len(wyb)), 100 * czulosc[wyb], color=["#c0504d"] * n + ["#4f81bd"] * n)
    ax.set_yticks(range(len(wyb)), [slowa[k] for k in wyb], fontsize=8)
    ax.axvline(100 * czulosc.mean(), color="k", ls="--", lw=1, label=f"średnia czułość {czulosc.mean():.1%}")
    ax.set(title=f"{n} słów o najniższej i {n} o najwyższej czułości (test)", xlabel="czułość", xlim=(0, 100))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "czulosc_slow.png", dpi=150)
    plt.close(fig)

    # pewność modelu przy trafieniach i pomyłkach
    ok = w["pred"] == w["praw"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(w["pewn"][ok], bins=20, range=(0, 1), alpha=0.7, label="trafienia")
    ax.hist(w["pewn"][~ok], bins=20, range=(0, 1), alpha=0.7, label="pomyłki")
    ax.set(title="Pewność modelu (największe prawdopodobieństwo)", xlabel="pewność", ylabel="liczba nagrań")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "pewnosc.png", dpi=150)
    plt.close(fig)

    wyniki = dict(info, test_nagran=len(ok), test_strata=round(w["strata"], 4), test_top1=round(w["top1"], 4),
                  test_top5=round(w["top5"], 4), test_WER=round(1 - w["top1"], 4),
                  test_czulosc_makro=round(float(czulosc.mean()), 4),
                  test_precyzja_makro=round(float(precyzja.mean()), 4), test_F1_makro=round(float(f1.mean()), 4),
                  sr_pewnosc_trafien=round(float(w["pewn"][ok].mean()), 4) if ok.any() else None,
                  sr_pewnosc_pomylek=round(float(w["pewn"][~ok].mean()), 4) if (~ok).any() else None,
                  slowo_najwyzsza_czulosc=slowa[kol[-1]], slowo_najnizsza_czulosc=slowa[kol[0]],
                  najczestsza_pomylka=f"{pomylki[0][1]} -> {pomylki[0][2]} ({pomylki[0][0]}x)" if pomylki else None)
    (out / "podsumowanie.json").write_text(json.dumps(wyniki, indent=2, ensure_ascii=False), encoding="utf-8")
    tekst = "\n".join(f"{k:28s} {v}" for k, v in wyniki.items())
    (out / "podsumowanie.txt").write_text(tekst + "\n", encoding="utf-8")
    print("\n" + tekst)


def main():
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

    ds = {c: LRWDataset(lrw, slowa, c, augmentacja=AUGMENTACJA and c == "train") for c in ("train", "val", "test")}
    dl = {c: torch.utils.data.DataLoader(ds[c], batch_size=BATCH, shuffle=c == "train", num_workers=WORKERS,
                                         pin_memory=True, drop_last=c == "train", persistent_workers=WORKERS > 0)
          for c in ds}
    print(f"{len(slowa)} słów | nagrania: trening {len(ds['train'])}, walidacja {len(ds['val'])}, "
          f"test {len(ds['test'])}")

    model = ModelWideo(len(slowa)).to(dev)
    n_param = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
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
    print(f"Parametry modelu: {n_param / 1e6:.1f} mln, epok: {EPOKI}, paczka: {BATCH}\n")

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

    print("Test najlepszego modelu...")
    model.load_state_dict(torch.load(out / "model_najlepszy.pt", map_location=dev))
    test = jedna_epoka(model, dl["test"], dev, strata)
    best = max(hist, key=lambda x: x["top1_val"])
    info = dict(nazwa=NAZWA, karta=karta, slow=len(slowa), nagran_trening=len(ds["train"]),
                nagran_walidacja=len(ds["val"]), parametry_mln=round(n_param / 1e6, 2), epok=len(hist),
                paczka=BATCH, lr=LR, augmentacja=AUGMENTACJA, najlepsza_epoka=best["epoka"],
                walidacja_top1=round(best["top1_val"], 4), walidacja_top5=round(best["top5_val"], 4),
                trening_top1_ostatnia=round(hist[-1]["top1_tr"], 4),
                sr_czas_epoki_min=round(float(np.mean([x["czas_min"] for x in hist])), 2),
                laczny_czas_h=round(sum(x["czas_min"] for x in hist) / 60, 2),
                max_pamiec_gb=round(max(x["pamiec_gb"] for x in hist), 2))
    raport_testu(test, slowa, out, info)
    print(f"\nWszystko zapisane w {out}")


if __name__ == "__main__":
    main()
