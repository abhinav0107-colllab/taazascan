"""TaazaScan proof-of-concept: ripening physics simulator + shelf-life models + store policy simulator.

Everything here is SIMULATED with stated assumptions. It validates the pipeline and the
decision layer; real accuracy must be measured on real multispectral captures (finale).

Physics model (per fruit):
  internal degradation D(t) in [0, 1]; visible spoilage when D >= 0.70
  dD/dt = k * Q10 ** ((T - 25) / 10),  Q10 = 2.5   (temperature-driven ripening)
  hidden handling bruise (25% of fruits): +0.12 D immediately and rate x1.4
Sensors:
  RGB colour/texture  -> only respond once D > ~0.5 (what the eye / normal camera sees)
  NIR bands (730/850/940 nm) -> respond to D from the start (water loss, cell-wall breakdown)
                                 and show sub-surface bruises
"""
from __future__ import annotations

import numpy as np

FRUITS = {            # name: (base days to visible spoilage at 25 C after ripe arrival, price INR/kg, weight kg)
    "banana": (4.0, 50, 0.15),
    "tomato": (6.0, 40, 0.10),
    "mango":  (5.0, 120, 0.30),
    "guava":  (4.0, 80, 0.20),
}
VISIBLE = 0.70
Q10 = 2.5


def make_fruits(n, rng):
    kinds = rng.choice(list(FRUITS), size=n)
    base = np.array([FRUITS[k][0] for k in kinds])
    k = (VISIBLE / base) * rng.lognormal(0, 0.25, n)          # natural variability
    d0 = rng.uniform(0.0, 0.35, n)                             # arrival state (field + transport)
    bruise = rng.random(n) < 0.25
    d0 = d0 + 0.12 * bruise
    k = k * np.where(bruise, 1.4, 1.0)
    temp_hist = rng.uniform(22, 36, n)                         # mean transport temperature (logger)
    d0 = d0 + 0.03 * (temp_hist - 25) / 10 * rng.uniform(0.5, 1.5, n)   # hot transport ages fruit
    d0 = np.clip(d0, 0, 0.6)
    return kinds, k, d0, bruise, temp_hist


def true_remaining_days(k, d, store_temp=25.0):
    rate = k * Q10 ** ((store_temp - 25) / 10)
    return np.maximum(0, (VISIBLE - d) / rate)


def observe(kinds, d, bruise, temp_hist, rng):
    """Feature vectors a camera system would produce for each fruit."""
    n = len(d)
    vis = np.clip((d - 0.5) / 0.2, 0, 1)                      # surface change only near the end
    rgb = np.stack([
        0.6 - 0.25 * vis + rng.normal(0, 0.04, n),             # hue shift
        0.5 - 0.20 * vis + rng.normal(0, 0.04, n),             # brightness
        0.1 + 0.30 * vis + rng.normal(0, 0.04, n),             # dark-spot area
        0.05 + 0.10 * bruise * vis + rng.normal(0, 0.03, n),   # bruise only visible late
    ], 1)
    nir = np.stack([
        0.8 - 0.45 * d + rng.normal(0, 0.035, n),              # 850 nm reflectance (water / cell walls)
        0.7 - 0.35 * d + rng.normal(0, 0.035, n),              # 940 nm (water band)
        0.4 + 0.30 * d + rng.normal(0, 0.035, n),              # 730/660 red-edge ratio (chlorophyll loss)
        0.05 + 0.14 * bruise * rng.uniform(0.3, 1.0, n) + rng.normal(0, 0.06, n),  # sub-surface bruise contrast in NIR (varies with severity)
    ], 1)
    kind_oh = np.stack([(kinds == f).astype(float) for f in FRUITS], 1)
    temp = ((temp_hist - 25) / 10)[:, None]
    return rgb, nir, kind_oh, temp


def build_dataset(n, seed):
    rng = np.random.default_rng(seed)
    kinds, k, d0, bruise, th = make_fruits(n, rng)
    # scan each fruit at a random day of its life before visible spoilage
    life0 = true_remaining_days(k, d0)
    t = rng.uniform(0, np.maximum(life0 - 0.05, 0.01))
    d = d0 + k * t
    y = true_remaining_days(k, d)
    rgb, nir, oh, temp = observe(kinds, d, bruise, th, rng)
    return dict(kinds=kinds, rgb=rgb, nir=nir, oh=oh, temp=temp, y=y, bruise=bruise, d=d)


FEATURE_SETS = {
    "RGB camera only": lambda D: np.hstack([D["rgb"], D["oh"]]),
    "RGB + NIR (multispectral)": lambda D: np.hstack([D["rgb"], D["nir"], D["oh"]]),
    "TaazaScan: RGB + NIR + temperature history": lambda D: np.hstack([D["rgb"], D["nir"], D["oh"], D["temp"]]),
}
