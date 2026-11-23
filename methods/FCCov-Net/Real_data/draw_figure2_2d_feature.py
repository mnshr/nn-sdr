import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data" / "3d_graph2"
DEFAULT_METADATA_DIR = ROOT / "data" / "jaffe"
DEFAULT_CHECKPOINT = ROOT / "data" / "2d_ckpt" / "mccov-H_d=2.pth"
DEFAULT_OUTPUT = ROOT / "figures" / "2d_feature.png"
DEFAULT_DI_IMAGE = "KR.DI3.88.tiff"
WHATEVER_FINAL_DIM1 = 12
WHATEVER_FINAL_DIM2 = 12


class DCNN(nn.Module):
    def __init__(self, p_dim):
        super(DCNN, self).__init__()
        self.p_dim = p_dim

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # Layer 1
            nn.ReLU(),  # Layer 2
            nn.MaxPool2d(kernel_size=2, stride=2),  # Layer 3
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),  # Layer 4
            nn.ReLU(),  # Layer 5
            nn.MaxPool2d(kernel_size=2, stride=2),  # Layer 6
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1),  # Layer 7
            nn.ReLU(),  # Layer 8
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # Layer 9
            nn.ReLU(),  # Layer 10
            nn.MaxPool2d(kernel_size=2, stride=2),  # Layer 11
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Layer 12
            nn.ReLU(),  # Layer 13
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # Layer 14
            nn.ReLU(),  # Layer 15
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256 * WHATEVER_FINAL_DIM1 * WHATEVER_FINAL_DIM2, 2000),  # Layer 16
            nn.ReLU(),  # Layer 17
            nn.Dropout(0.3),
            nn.Linear(2000, self.p_dim),  # Layer 18
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CustomNet(nn.Module):
    def __init__(
        self,
        architecture="DCNN",
        optimizer_name="SGD",
        blambda=2e-6,
        device="cpu",
        output_dim=2,
        batch_size=64,
        iter_num=500,
        dist_mode=3,
        lr=0.01,
    ):
        super(CustomNet, self).__init__()
        self.device = torch.device(device)
        self.output_dim = output_dim
        self.blambda = blambda
        self.batch_size = batch_size
        self.iter_num = iter_num
        self.dist_mode = dist_mode
        self.lr = lr
        self.arch = architecture

        if architecture != "DCNN":
            raise ValueError("This reproduction script only implements the notebook DCNN branch.")

        self.model = DCNN(self.output_dim)
        linear_1_params = []
        linear_2_params = []

        for name, param in self.model.named_parameters():
            if "classifier.1" in name:
                linear_1_params.append(param)
            elif "classifier.4" in name:
                linear_2_params.append(param)

        self.optimizer = optim.SGD(
            [
                {"params": linear_1_params, "weight_decay": 0.0005},
                {"params": linear_2_params, "weight_decay": 0.0025},
            ],
            lr=self.lr,
            momentum=0.9,
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[5000, 18000], gamma=0.1
        )
        self.model.to(self.device)

    def forward(self, x):
        return self.model(x)

    def transform(self, data):
        self.eval()
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)

        outputs = []
        with torch.no_grad():
            for start in range(0, data.size(0), self.batch_size):
                batch = data[start : start + self.batch_size].to(self.device)
                outputs.append(self.forward(batch).cpu())
        return torch.cat(outputs, dim=0).numpy().astype(np.float32)


def find_seven(arr, seven=6):
    sorted_indices = np.argsort(arr)
    if len(arr) < seven:
        raise ValueError(f"Need at least {seven} values, got {len(arr)}")
    idx = sorted_indices[seven - 1]
    return arr[idx], idx


def load_embedding(data_dir, checkpoint, device):
    train_data = np.load(data_dir / "m_tr_data.npy")
    train_label = np.load(data_dir / "tr_label.npy")

    model = CustomNet(
        architecture="DCNN",
        optimizer_name="SGD",
        iter_num=500,
        batch_size=64,
        lr=0.01,
        blambda=2e-6,
        dist_mode=3,
        device=device,
        output_dim=2,
    )
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.transform(train_data), train_label, train_data


def resolve_train_index_from_jaffe_filename(filename, metadata_dir):
    if not filename:
        return None

    parts = Path(filename).stem.split(".")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse JAFFE filename: {filename}")
    target_id = int(parts[-1])

    label_table = np.genfromtxt(
        metadata_dir / "label.txt", skip_header=2, usecols=range(7)
    )
    label_indices = label_table[:, 0].astype(int)
    rows = label_table[:, 1:]
    matches = np.flatnonzero(label_indices == target_id)
    if len(matches) != 1:
        raise ValueError(f"Expected one label.txt row for index {target_id}, got {len(matches)}")
    target_counts = rows[matches[0]]

    counts = np.load(metadata_dir / "counts.npy")
    data_hits = np.flatnonzero(np.all(np.isclose(counts, target_counts), axis=1))
    if len(data_hits) != 1:
        raise ValueError(
            f"Expected one counts.npy match for {filename}, got {len(data_hits)}"
        )

    train_indices, _ = train_test_split(
        np.arange(len(counts)), test_size=0.2, random_state=42
    )
    train_hits = np.flatnonzero(train_indices == data_hits[0])
    if len(train_hits) != 1:
        raise ValueError(f"{filename} is not in the training split")
    return int(train_hits[0])


def draw_figure(x_train, y_train, m_tr_data, output_path, di_rank, di_global_index):
    anti_emotion = {0: "HA", 1: "SA", 2: "SU", 3: "AN", 4: "DI", 5: "FE", 6: "NE"}
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    markers = ["o", "s", "^", "D", "v", "P", "*"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

    temp = np.squeeze(m_tr_data, axis=1)
    xs = []
    ys = []
    images = []

    for i in range(7):
        mask = y_train == i
        img = temp[mask, :, :]
        xp = x_train[mask, 0]
        yp = x_train[mask, 1]
        ax.scatter(
            xp,
            yp,
            color=colors[i],
            marker=markers[i],
            label=f"Class:{anti_emotion[i]}",
            alpha=0.7,
        )

        if i == 5:
            value, idx = find_seven(xp, 10)
        elif i == 4 and di_global_index is not None:
            global_indices = np.flatnonzero(mask)
            matches = np.flatnonzero(global_indices == di_global_index)
            if len(matches) != 1:
                raise ValueError(f"DI global index {di_global_index} is not in class DI")
            idx = int(matches[0])
            value = xp[idx]
        elif i == 4:
            value, idx = find_seven(xp, di_rank)
        else:
            value, idx = find_seven(xp, 6)
        xs.append(value)
        ys.append(yp[idx])
        images.append(img[idx])

    for image, x0, y0 in zip(images, xs, ys):
        im = OffsetImage(image, zoom=0.4, cmap="gray")
        ab = AnnotationBbox(im, (x0 + 0.5, y0 + 0.8), frameon=False)
        ax.add_artist(ab)

    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce the JAFFE 2D feature scatter plot with embedded images."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device. Use cpu for reproducibility, or cuda:0/cuda:1/etc. Default: cpu.",
    )
    parser.add_argument(
        "--di-rank",
        type=int,
        default=7,
        help=(
            "Fallback DI annotation rank when --di-image is empty. "
            "The pasted notebook cell uses 7."
        ),
    )
    parser.add_argument(
        "--di-image",
        default=DEFAULT_DI_IMAGE,
        help=(
            "JAFFE filename used for the DI embedded face. "
            "Default KR.DI3.88.tiff matches the archived original PNG. "
            "Pass an empty string to use --di-rank instead."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir.resolve()
    metadata_dir = args.metadata_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not metadata_dir.exists():
        raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    x_train, y_train, m_tr_data = load_embedding(data_dir, checkpoint, args.device)
    di_global_index = resolve_train_index_from_jaffe_filename(args.di_image, metadata_dir)
    draw_figure(x_train, y_train, m_tr_data, output, args.di_rank, di_global_index)
    print(f"Saved figure to: {output}")


if __name__ == "__main__":
    main()
