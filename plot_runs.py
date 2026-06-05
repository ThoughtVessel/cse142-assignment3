import re
import sys
import matplotlib.pyplot as plt


def parse_log(path):
    steps, train, val = [], [], []
    for line in open(path):
        m = re.search(r'step\s+(\d+).*train loss ([\d.]+).*val loss ([\d.]+)', line)
        if m:
            steps.append(int(m.group(1)))
            train.append(float(m.group(2)))
            val.append(float(m.group(3)))
    return steps, train, val


if len(sys.argv) < 2:
    print("Usage: python plot_runs.py run1.log run2.log ...")
    sys.exit(1)

paths = sys.argv[1:]
n = len(paths)
cols = 2
rows = (n + 1) // 2
fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
axes = axes.flatten() if n > 1 else [axes]

for ax, path in zip(axes, paths):
    name = path.split("/")[-1].replace(".log", "")
    steps, train, val = parse_log(path)
    ax.plot(steps, train, linestyle="--", color="steelblue", label="train")
    ax.plot(steps, val, color="steelblue", label="val")
    final_train = train[-1] if train else float("nan")
    final_val = val[-1] if val else float("nan")
    ax.set_ylim(min(train + val) * 0.98, 2.5)
    ax.set_title(f"{name}\nfinal train: {final_train:.4f} | final val: {final_val:.4f}")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()

for ax in axes[n:]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig("curves.png")
print("Saved curves.png")
