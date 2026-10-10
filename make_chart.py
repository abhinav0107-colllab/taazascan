import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
r = json.load(open("metrics.json"))
st = r["store"]
labels = ["Today\n(FIFO)", "RGB camera\n+ smart decisions", "TaazaScan\n(multispectral)"]
vals = [v["waste_pct"] * 100 for v in st.values()]
fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
b = ax.bar(labels, vals, color=["#DC2626", "#94A3B8", "#0F766E"], width=0.55)
for rect, v in zip(b, vals):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.2, f"{v:.1f}%", ha="center", fontsize=10)
ax.set_ylabel("% of fruit received that is thrown away"); ax.set_title("30-day store simulation at 30 °C (Indian ambient)", fontsize=10)
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.25)
plt.tight_layout(); plt.savefig("waste.png"); print("saved")
