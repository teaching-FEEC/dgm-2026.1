import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

RESULTS_DIR = Path(__file__).parent / "results"
RECURRENT_JSON = RESULTS_DIR / "recurrent_errors.json"
# Insert below the FakeClue test dataset path
IMAGES_BASE = Path("~/Datasets/FakeClue_Data/test")

with open(RECURRENT_JSON) as f:
    recurrent = json.load(f)

entries: list[tuple[str, str]] = []  
for category, items in recurrent.items():
    for item in items:
        entries.append((category, item["image"]))

BATCH = 4
current = [0]  

def show_batch(start: int) -> None:
    plt.clf()
    batch = entries[start : start + BATCH]
    n = len(batch)
    cols = min(n, 2)
    rows = (n + 1) // 2

    for i, (category, rel_path) in enumerate(batch):
        ax = plt.subplot(rows, cols, i + 1)
        img_path = IMAGES_BASE / rel_path
        if img_path.exists():
            img = mpimg.imread(img_path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "Image not found", ha="center", va="center",
                    transform=ax.transAxes, color="red")
        ax.set_title(rel_path, fontsize=6, wrap=True)
        ax.axis("off")

    total = len(entries)
    end = min(start + BATCH, total)
    plt.suptitle(
        f"[{start + 1}–{end} / {total}]  Space = next  |  Esc = quit",
        fontsize=9,
    )
    plt.tight_layout()
    plt.draw()


def on_key(event) -> None:
    if event.key == " ":
        next_start = current[0] + BATCH
        if next_start >= len(entries):
            print("No more images.")
            return
        current[0] = next_start
        show_batch(current[0])
    elif event.key == "escape":
        plt.close("all")


fig = plt.figure(figsize=(12, 7))
fig.canvas.mpl_connect("key_press_event", on_key)
show_batch(0)
plt.show()
