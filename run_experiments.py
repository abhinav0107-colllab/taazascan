"""Reproduce every TaazaScan number:  python run_experiments.py   (about 1 minute)"""
import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, roc_auc_score

from taazascan.sim import (FEATURE_SETS, FRUITS, build_dataset, make_fruits, observe,
                           true_remaining_days)

train, test = build_dataset(12000, seed=1), build_dataset(5000, seed=2)
models, res = {}, {"shelf_life": {}, "hidden_bruise": {}, "store": {}}

# ---------------------------------------------------------------- 1. remaining-shelf-life models
pre_visual = (test["d"] < 0.5)                          # nothing visible to eye / RGB yet
soon = test["y"] <= 1.5                                 # will spoil within 1.5 days
for name, f in FEATURE_SETS.items():
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0).fit(f(train), train["y"])
    pred = m.predict(f(test))
    models[name] = m
    flag = pred <= 1.5
    res["shelf_life"][name] = dict(
        mae_days=float(mean_absolute_error(test["y"], pred)),
        early_warning=float(flag[soon & pre_visual].mean()),          # caught before visible
        false_flag=float(flag[test["y"] > 3].mean()),                 # flagged although > 3 days left
    )

# ---------------------------------------------------------------- 2. hidden bruise detection (before visible)
for name, cols in {"RGB camera only": ["rgb"], "NIR (multispectral)": ["rgb", "nir"]}.items():
    X = lambda D: np.hstack([D[c] for c in cols] + [D["oh"]])
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X(train), train["bruise"])
    p = clf.predict_proba(X(test))[:, 1]
    res["hidden_bruise"][name] = dict(auc_pre_visual=float(roc_auc_score(test["bruise"][pre_visual], p[pre_visual])))

# ---------------------------------------------------------------- 3. 30-day store simulation
def store(policy, model_name=None, days=30, arrivals=200, demand=175, markdown_cap=25, juice_cap=15, store_temp=30.0, seed=11):
    rng = np.random.default_rng(seed)
    inv = dict(kind=np.array([], dtype="<U6"), k=np.array([]), d=np.array([]), b=np.array([], bool),
               th=np.array([]), day=np.array([]))
    received = wasted = 0.0
    value_full = value_got = 0.0
    for day in range(days):
        kinds, k, d0, b, th = make_fruits(arrivals, rng)
        for key, v in zip(["kind", "k", "d", "b", "th", "day"], [kinds, k, d0, b, th, np.full(arrivals, day)]):
            inv[key] = np.concatenate([inv[key], v])
        w = np.array([FRUITS[x][2] for x in kinds]); pr = np.array([FRUITS[x][1] for x in kinds])
        received += w.sum(); value_full += (w * pr).sum()
        n = len(inv["d"])
        wt = np.array([FRUITS[x][2] for x in inv["kind"]]); price = np.array([FRUITS[x][1] for x in inv["kind"]]) * wt
        if policy == "fifo":
            order = np.argsort(inv["day"], kind="stable"); pred = None
        else:
            rgb, nir, oh, temp = observe(inv["kind"], inv["d"], inv["b"], inv["th"], rng)
            D = dict(rgb=rgb, nir=nir, oh=oh, temp=temp)
            pred = models[model_name].predict(FEATURE_SETS[model_name](D))
            order = np.argsort(pred)                                  # FEFO: first-expiring first out
        sold = np.zeros(n, bool)
        got = np.zeros(n)
        if pred is not None:
            # sell-through aware: only discount fruit that will NOT sell at full price before it spoils
            pos = np.empty(n); pos[order] = np.arange(n)
            will_sell_in = pos / demand                                 # days until it reaches the front of the FEFO queue
            at_risk = pred < will_sell_in + 0.5
            juice = np.where(at_risk & (pred <= 0.5))[0]                # route to juice / cut-fruit counter
            juice = juice[np.argsort(pred[juice])][:juice_cap]
            sold[juice] = True; got[juice] = 0.5 * price[juice]
            mark = np.where(at_risk & ~sold)[0]                         # same-day 30% markdown, limited buyers
            mark = mark[np.argsort(pred[mark])][:markdown_cap]
            sold[mark] = True; got[mark] = 0.7 * price[mark]
        regular = [i for i in order if not sold[i]][:demand]
        sold[regular] = True; got[regular] = price[regular]
        value_got += got.sum()
        keep = ~sold
        for key in inv: inv[key] = inv[key][keep]
        inv["d"] = inv["d"] + inv["k"] * 2.5 ** ((store_temp - 25) / 10)   # one day at store temperature
        spoiled = inv["d"] >= 0.70
        wasted += np.array([FRUITS[x][2] for x in inv["kind"][spoiled]]).sum()
        for key in inv: inv[key] = inv[key][~spoiled]
    return dict(waste_pct=wasted / received, value_recovered_pct=value_got / value_full)

res["store"]["Today: FIFO, throw away when visibly spoiled"] = store("fifo")
res["store"]["RGB camera: FEFO + markdown"] = store("ml", "RGB camera only")
res["store"]["TaazaScan: multispectral FEFO + markdown + juice route"] = store("ml", "TaazaScan: RGB + NIR + temperature history")

for sec, d in res.items():
    print(f"== {sec}")
    for k, v in d.items():
        print(f"  {k:55s} " + "  ".join(f"{m}={x:.3f}" for m, x in v.items()))
json.dump(res, open("results/metrics.json", "w"), indent=2)
