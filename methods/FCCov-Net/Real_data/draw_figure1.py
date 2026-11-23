import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import make_interp_spline


CURRENT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(CURRENT_DIR, "data", "jaffe")
IMAGE_DIR = DATA_DIR
LABEL_PATH = os.path.join(DATA_DIR, "label.txt")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "figures")

SELECTED_FILENAMES = [
    "TM.HA2.181.tiff",
    "KA.DI2.43.tiff",
    "KM.SU2.15.tiff",
    "KR.AN1.83.tiff",
]


def load_images_from_folder(folder, selected_filenames):
    images_data = []

    for filename in selected_filenames:
        if filename.endswith(".tiff"):
            name_parts = filename.split(".")
            name = name_parts[0]
            label_number = name_parts[1]
            index = name_parts[2]

            label = label_number[:-1]
            number = label_number[-1]

            image_path = os.path.join(folder, filename)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Missing image file: {image_path}")

            with Image.open(image_path) as img:
                images_data.append(
                    {
                        "image": img.copy().convert("L"),
                        "name": name,
                        "label": label,
                        "number": number,
                        "index": index,
                        "filename": filename,
                    }
                )

    return images_data


def load_raw_data_and_distribution():
    images = load_images_from_folder(IMAGE_DIR, SELECTED_FILENAMES)

    label_table = pd.read_csv(LABEL_PATH, sep=r"\s+", skiprows=2, header=None)
    label_table.columns = ["index", "HAP", "SAD", "SUR", "ANG", "DIS", "FEA", "PIC"]
    label_table.drop("PIC", axis=1, inplace=True)
    label_table.set_index("index", inplace=True)

    images_data = []
    labels = []
    counts = []
    filenames = []

    for image_info in images:
        img_index = int(image_info["index"])
        if img_index in label_table.index:
            images_data.append(np.array(image_info["image"]))
            labels.append(image_info["label"])
            counts.append(np.array(label_table.loc[img_index], dtype=float))
            filenames.append(image_info["filename"])

    data = np.array(images_data)
    label = np.array(labels)
    counts = np.array(counts)

    dis_sum = counts.sum(axis=1)[:, np.newaxis]
    dis = np.tile(dis_sum, (1, 6))
    distribution = counts / dis
    return data, label, counts, distribution, filenames


def resolve_save_path(save_path):
    if save_path is None:
        return None
    if not os.path.splitext(save_path)[1]:
        save_path = f"{save_path}.png"
    if not os.path.isabs(save_path):
        save_path = os.path.join(OUTPUT_DIR, save_path)
    return save_path


def plot_image_and_distribution(data, distribution, save_path=None):
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    ax0 = plt.subplot(gs[0])
    ax0.imshow(data, cmap="gray")
    ax0.axis("off")

    ax1 = plt.subplot(gs[1])
    categories = ["HAP", "SAD", "SUR", "ANG", "DIS", "FER"]
    bars = ax1.bar(np.arange(len(distribution)), distribution, width=0.4)
    ax1.set_xticks(np.arange(len(categories)))
    ax1.set_xticklabels(categories)

    midpoints = [bar.get_x() + bar.get_width() / 2 for bar in bars]
    heights = [bar.get_height() for bar in bars]

    x_new = np.linspace(min(midpoints), max(midpoints), 300)
    spline = make_interp_spline(midpoints, heights, k=2)
    heights_smooth = spline(x_new)

    ax1.plot(x_new, heights_smooth, "k--")

    plt.tight_layout()

    save_path = resolve_save_path(save_path)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    plt.close(fig)


def plot_distribution(ax, distribution):
    categories = ["HAP", "SAD", "SUR", "ANG", "DIS", "FER"]
    bars = ax.bar(np.arange(len(distribution)), distribution, width=0.4)
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories)

    midpoints = [bar.get_x() + bar.get_width() / 2 for bar in bars]
    heights = [bar.get_height() for bar in bars]
    x_new = np.linspace(min(midpoints), max(midpoints), 300)
    spline = make_interp_spline(midpoints, heights, k=2)
    ax.plot(x_new, spline(x_new), "k--")


def plot_image_grid(data, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for ax, image in zip(axes.ravel(), data):
        ax.imshow(image, cmap="gray")
        ax.axis("off")

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    save_path = resolve_save_path(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    print(f"Saved figure to: {save_path}")
    plt.close(fig)


def plot_distribution_grid(distribution, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for ax, dist in zip(axes.ravel(), distribution):
        plot_distribution(ax, dist)

    fig.tight_layout()
    save_path = resolve_save_path(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {save_path}")
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data, label, counts, distribution, filenames = load_raw_data_and_distribution()
    for j, filename in enumerate(filenames):
        output_name = f"{os.path.splitext(filename)[0]}_distribution.png"
        print(f"Using image {j}: {filename} label={label[j]} counts={counts[j]}")
        plot_image_and_distribution(
            data[j],
            distribution[j],
            output_name,
        )

    plot_image_grid(data, "image_combination.png")
    plot_distribution_grid(distribution, "dis_combination.png")


if __name__ == "__main__":
    main()
