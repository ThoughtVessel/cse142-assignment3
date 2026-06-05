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

for path in sys.argv[1:]:
    name = path.replace(".log", "")
    steps, train, val = parse_log(path)
    plt.plot(steps, train, linestyle="--", label=f"{name} train")
    plt.plot(steps, val, label=f"{name} val")

plt.xlabel("step")
plt.ylabel("loss")
plt.legend()
plt.tight_layout()
plt.savefig("curves.png")
print("Saved curves.png")
