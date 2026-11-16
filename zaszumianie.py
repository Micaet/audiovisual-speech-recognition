"""
Zaszumianie dźwięku 16 kHz do treningu, testu i interfejsu graficznego.

Rodzaje szumu:
    bialy, rozowy, brazowy - generowane matematycznie
    babble                 - gwar: suma kilku nagrań mowy podanych w `glosy`

SNR jest liczony na fragmencie ze słowem (granice z pliku .npz), a nie na całym nagraniu,
bo cisza przed i po słowie zaniżałaby moc mowy.

Użycie:
    import zaszumianie as zs
    rng = np.random.default_rng(zs.ziarno("ABOUT_00001", "babble", 0))   # ten sam szum przy każdym wywołaniu
    y = zs.zaszum(audio, "babble", 0, rng, granice=(poczatek_slowa, koniec_slowa), glosy=glosy)
"""
import hashlib

import numpy as np

SR, FPS = 16000, 25


def ziarno(*czesci):
    """Stała liczba z dowolnych napisów i liczb: ten sam opis daje zawsze ten sam szum."""
    return int(hashlib.md5("|".join(map(str, czesci)).encode()).hexdigest()[:8], 16)


def moc(x):
    return float(np.mean(np.asarray(x, np.float64) ** 2)) + 1e-12


def region_mowy(audio, granice=None):
    """(start, stop) fragmentu z mową w próbkach. granice = (poczatek_slowa, koniec_slowa) w klatkach 25 kl./s.
    Bez granic: ramki najwyżej 20 dB ciszej od najgłośniejszej."""
    if granice is not None:
        a = int(max(0.0, float(granice[0])) * SR / FPS)
        b = int(min(len(audio), float(granice[1]) * SR / FPS))
        if b - a > SR // 20:
            return a, b
    n = SR // 50
    k = len(audio) // n
    if k == 0:
        return 0, len(audio)
    e = 10 * np.log10((np.asarray(audio[:k * n], np.float64).reshape(k, n) ** 2).mean(1) + 1e-12)
    akt = np.where(e > e.max() - 20)[0]
    return int(akt[0] * n), int((akt[-1] + 1) * n)


def szum_kolorowy(n, rodzaj, rng):
    """Szum biały (równa moc w całym paśmie), różowy (moc ~ 1/f) albo brązowy (moc ~ 1/f²)."""
    w = rng.standard_normal(n)
    if rodzaj == "bialy":
        return w.astype(np.float32)
    X = np.fft.rfft(w)
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    X = X / (np.sqrt(f) if rodzaj == "rozowy" else f)
    y = np.fft.irfft(X, n)
    return (y / (y.std() + 1e-12)).astype(np.float32)


def babble(glosy, n, rng, k=6):
    """Gwar: suma k losowych nagrań z tablicy glosy (liczba_nagrań, próbki), każde o tej samej głośności."""
    g = glosy[rng.choice(len(glosy), k, replace=False), :n]
    g = g / (g.std(1, keepdims=True) + 1e-6)
    return g.sum(0)


def dodaj_szum(mowa, szum, snr_db, region=None):
    """Miesza mowę z szumem tak, żeby na fragmencie `region` SNR wynosił dokładnie snr_db.
    Gdy wynik przekracza zakres -1..1, ścisza całość (SNR się nie zmienia)."""
    mowa = np.asarray(mowa, np.float32)
    szum = np.asarray(szum[:len(mowa)], np.float32)
    a, b = region or (0, len(mowa))
    y = mowa + szum * np.sqrt(moc(mowa[a:b]) / (moc(szum[a:b]) * 10 ** (snr_db / 10)))
    szczyt = np.abs(y).max()
    if szczyt > 0.99:
        y = y * (0.99 / szczyt)
    return y.astype(np.float32)


def zaszum(audio, rodzaj, snr_db, rng, granice=None, glosy=None, glosy_babble=6):
    """Dźwięk z dodanym szumem danego rodzaju przy zadanym SNR. Dla babble trzeba podać glosy."""
    if rodzaj == "babble":
        szum = babble(glosy, len(audio), rng, glosy_babble)
    else:
        szum = szum_kolorowy(len(audio), rodzaj, rng)
    return dodaj_szum(audio, szum, snr_db, region_mowy(audio, granice))
