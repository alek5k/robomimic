from dataclasses import dataclass
from enum import Enum
from glob import glob
from pathlib import Path
import os

import draccus
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class DatasetType(str, Enum):
    UNKNOWN = "Unknown"
    DEMO = "Demo"
    BASELINE = "Baseline"
    INFERENCE = "Inference"


@dataclass
class DatasetLocator:
    path: str
    label: str | None = None
    dataset_type: DatasetType = DatasetType.UNKNOWN


@dataclass
class ParsedDatasetInfo:
    path: str
    is_demo_dataset: bool
    task_name: str
    dataset_type: DatasetType = DatasetType.UNKNOWN
    short_label: str | None = None
    step_counts: list[float] | None = None
    success_rate: float | None = None
    W_episode: float | None = None
    W_total: float | None = None


@dataclass
class AnalysisConfig:
    num_episodes: int = 300
    task_name: str = "can"
    histogram_bins: int = 81
    include_rankings: bool = False
    include_demo_dataset: bool = False
    generate_tex_tables: bool = False

    h_space: float = 0.05
    w_space: float = 0.05
    subplot_width_scale: float = 7.0
    subplot_height_scale: float = 1.8
    use_single_overlay_legend: bool = False
    hide_second_col_y_tick_labels: bool = False
    axis_label_font_size: int | None = None
    overlay_legend_font_size: int | None = None
    demo_legend_label: str = "Demonstration Dataset"


TASK_NAMES = {"can", "square", "lift", "tool_hang", "transport"}

font_size = 18
legend_size = 14
tick_label_size = 14

matplotlib.rcParams.update({
    "font.size": font_size,
    "axes.titlesize": font_size,
    "axes.labelsize": font_size,
    "xtick.labelsize": tick_label_size,
    "ytick.labelsize": tick_label_size,
    "legend.fontsize": legend_size,
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{times}",
    "font.family": "serif",
    "axes.unicode_minus": False,
})


#####################
## DEFAULT CONFIGS ##
#####################

def default_robomimic_mh_datasets(task_name: str) -> list[DatasetLocator]:
    return [
        DatasetLocator(path=f"datasets/{task_name}/mh/image_v15.hdf5", label="Demo", dataset_type=DatasetType.DEMO),
        DatasetLocator(path=f"rollouts/{task_name}/bc/*.hdf5", label="BC", dataset_type=DatasetType.BASELINE),
        # DatasetLocator(path=f"rollouts/{task_name}/bc_rnn/*.hdf5", label="BC-RNN", dataset_type=DatasetType.BASELINE),
        DatasetLocator(path=f"rollouts/{task_name}/hbc/*.hdf5", label="HBC", dataset_type=DatasetType.BASELINE),
        DatasetLocator(path=f"rollouts/{task_name}/diffusion_policy/*.hdf5", label="DP", dataset_type=DatasetType.BASELINE),
        DatasetLocator(path=f"rollouts/{task_name}/tc_diffusion_policy/*.hdf5", label="TC-DP", dataset_type=DatasetType.INFERENCE),
    ]


def select_datasets_by_type(datasets: list[DatasetLocator], allowed_types: set[DatasetType]) -> list[DatasetLocator]:
    return [d for d in datasets if d.dataset_type in allowed_types]


def print_table(datasets: list[ParsedDatasetInfo]):
    from tabulate import tabulate

    rows = []
    headers = [
        "#", "label", "W_total", "W_episode", "success",
        "path",
    ]

    for i, d in enumerate(datasets):
        rows.append([
            i,
            d.short_label,
            d.W_total,
            d.W_episode,
            d.success_rate,
            d.path,
        ])

    print(tabulate(rows, headers=headers, tablefmt="github", floatfmt=".4f", missingval=""))


def extract_episode_lengths(dataset_path: str, num_episodes: int | None = None) -> list[int]:
    with h5py.File(dataset_path, "r") as f:
        demos = sorted(list(f["data"].keys()), key=lambda name: int(name.split("_")[-1]))
        if num_episodes is not None:
            demos = demos[:num_episodes]

        lengths = []
        for demo in demos:
            demo_group = f["data"][demo]
            length = demo_group.attrs.get("num_samples", None)
            if length is None:
                length = demo_group["actions"].shape[0]
            lengths.append(int(length))
    return lengths


def task_name_from_path(path: str) -> str:
    normalized = Path(path)
    for part in normalized.parts:
        if part in TASK_NAMES:
            return part

    raise ValueError(f"Could not infer task name from {path}")


@draccus.wrap()
def main(cfg: AnalysisConfig):
    task_name = cfg.task_name
    if task_name not in TASK_NAMES:
        raise ValueError(f"task_name must be one of {sorted(TASK_NAMES)}, got {task_name!r}")

    datasets = default_robomimic_mh_datasets(task_name)
    dataset_types = {DatasetType.DEMO, DatasetType.BASELINE, DatasetType.INFERENCE}
    parsed_datasets = parse_datasets(cfg, select_datasets_by_type(datasets, dataset_types))
    generate_graphs(cfg, parsed_datasets, save_prefix=f"robomimic_{task_name}_mh")

    if cfg.generate_tex_tables:
        print_table(sorted(parsed_datasets.values(), key=lambda item: item.W_total or 0.0))


def parse_datasets(cfg: AnalysisConfig, datasets: list[DatasetLocator]) -> dict[str, ParsedDatasetInfo]:
    parsed_datasets = {}

    for dataset in datasets:
        if dataset.dataset_type == DatasetType.DEMO:
            if not os.path.exists(dataset.path):
                print(f"Skipping missing demo dataset: {dataset.path}")
                continue

            print(f"Parsing dataset: {dataset.label or 'Demo'}")
            step_counts = extract_episode_lengths(dataset.path, cfg.num_episodes)
            task_name = task_name_from_path(dataset.path)
            parsed_datasets[dataset.path] = ParsedDatasetInfo(
                path=dataset.path,
                short_label=dataset.label or "Demo",
                is_demo_dataset=True,
                task_name=task_name,
                dataset_type=dataset.dataset_type,
                step_counts=step_counts,
                success_rate=100.0,
            )
            print("Demo episodes: {} ({})".format(len(step_counts), dataset.label or "Demo"))
            continue

        rollout_paths = sorted(glob(dataset.path))
        if not rollout_paths:
            print(f"Skipping missing rollout datasets: {dataset.path}")
            continue

        combined_step_counts = []
        task_name = None
        for rollout_path in rollout_paths:
            step_counts = extract_episode_lengths(rollout_path, cfg.num_episodes)
            if not step_counts:
                print(f"Skipping rollout dataset without episodes: {rollout_path}")
                continue

            if task_name is None:
                task_name = task_name_from_path(rollout_path)
            combined_step_counts.extend(step_counts)
            print(f"Parsing rollout dataset: {dataset.label or 'Rollout'} ({rollout_path})")

        if not combined_step_counts:
            print(f"Skipping rollout dataset without episodes: {dataset.path}")
            continue

        label = dataset.label or Path(rollout_paths[0]).parent.name
        parsed_datasets[dataset.path] = ParsedDatasetInfo(
            path=dataset.path,
            short_label=label,
            is_demo_dataset=False,
            task_name=task_name or cfg.task_name,
            dataset_type=dataset.dataset_type or DatasetType.UNKNOWN,
            step_counts=combined_step_counts,
        )
        print("Combined episodes: {} ({})".format(len(combined_step_counts), label))

    calculate_wasserstein_distances(parsed_datasets)
    return parsed_datasets


def calculate_wasserstein_distances(parsed_datasets: dict[str, ParsedDatasetInfo]):
    from scipy.stats import wasserstein_distance

    demonstration_dataset = next(d for d in parsed_datasets.values() if d.is_demo_dataset)
    demo_steps = np.asarray(demonstration_dataset.step_counts, dtype=float)

    for dataset_path, parsed_info in parsed_datasets.items():
        step_counts = np.asarray(parsed_info.step_counts, dtype=float)
        w_distance_step = wasserstein_distance(step_counts, demo_steps)
        parsed_datasets[dataset_path].W_episode = w_distance_step
        parsed_datasets[dataset_path].W_total = w_distance_step


def generate_graphs(cfg: AnalysisConfig, parsed_datasets: dict[str, ParsedDatasetInfo], save_prefix: str):
    output_dir = Path(__file__).resolve().parent / "analysis_temporal"
    output_dir.mkdir(parents=True, exist_ok=True)

    demonstration_dataset = next(d for d in parsed_datasets.values() if d.is_demo_dataset)
    step_count_arrays = [np.asarray(d.step_counts, dtype=float) for d in parsed_datasets.values()]
    combined_step_counts = np.concatenate(step_count_arrays)
    bins_step_counts = np.histogram_bin_edges(combined_step_counts, bins=cfg.histogram_bins)
    ref_step_hist, _ = np.histogram(demonstration_dataset.step_counts, bins=bins_step_counts)
    step_centers = 0.5 * (bins_step_counts[:-1] + bins_step_counts[1:])
    step_widths = np.diff(bins_step_counts)

    rows_to_plot = [
        (path, info)
        for path, info in parsed_datasets.items()
        if cfg.include_demo_dataset or not info.is_demo_dataset
    ]
    if not rows_to_plot:
        raise ValueError("No non-demo datasets to plot.")

    ncols = 1 + int(cfg.include_rankings)
    width_ratios = [1] + ([0.25] if cfg.include_rankings else [])
    fig, axes = plt.subplots(
        nrows=len(rows_to_plot),
        ncols=ncols,
        figsize=(cfg.subplot_width_scale * sum(width_ratios), cfg.subplot_height_scale * len(rows_to_plot)),
        sharex="col",
        sharey="col",
        gridspec_kw={"width_ratios": width_ratios},
    )
    axes = np.asarray(axes)
    if axes.ndim == 0:
        axes = axes.reshape(1, 1)
    elif axes.ndim == 1:
        axes = axes[:, np.newaxis]

    from matplotlib.patches import Patch

    cmap = plt.get_cmap("autumn")
    all_wd_totals = [d.W_total for d in parsed_datasets.values() if not d.is_demo_dataset]
    sorted_wd_totals = sorted(v for v in all_wd_totals if v is not None)
    axis_label_kwargs = {}
    if cfg.axis_label_font_size is not None:
        axis_label_kwargs["fontsize"] = cfg.axis_label_font_size

    for row, (_dataset_path, parsed_info) in enumerate(rows_to_plot):
        step_count_data = np.asarray(parsed_info.step_counts, dtype=float)
        label = parsed_info.short_label or parsed_info.path
        color = cmap(1.0 - row / max(len(rows_to_plot) - 1, 1))
        step_hist, _ = np.histogram(step_count_data, bins=bins_step_counts)

        stepcount_axis = axes[row, 0]
        stepcount_axis.grid(False)
        if cfg.hide_second_col_y_tick_labels:
            stepcount_axis.tick_params(axis="y", which="both", left=False, labelleft=False)

        stepcount_axis.bar(
            step_centers,
            step_hist,
            width=step_widths,
            color=color,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )

        mask = ref_step_hist != 0
        stepcount_axis.bar(
            step_centers[mask],
            ref_step_hist[mask],
            width=step_widths[mask],
            facecolor="none",
            edgecolor="blue",
        )

        wd_stepcount = parsed_info.W_episode or 0.0
        label_stepcount_wd = f"$W_{{episode}}={wd_stepcount:.2f}$"
        stepcount_axis.set_ylabel("Frequency", fontweight="bold", **axis_label_kwargs)
        stepcount_axis.legend(
            handles=[
                Patch(facecolor=color, edgecolor="black", alpha=0.7, label=f"{label}, {label_stepcount_wd}"),
                Patch(facecolor="none", edgecolor="blue", linewidth=1.5, label=cfg.demo_legend_label),
            ],
            loc="upper right",
        )

        if cfg.include_rankings:
            text_axis = axes[row, 1]
            text_axis.axis("off")
            rank = sorted_wd_totals.index(parsed_info.W_total) + 1 if parsed_info.W_total in sorted_wd_totals else 0
            text_axis.text(
                0.0,
                0.5,
                f"$W_{{episode}}$: {wd_stepcount:.2f}\nRank: {rank}",
                transform=text_axis.transAxes,
                ha="left",
                va="center",
                fontsize=10,
            )

        if row == len(rows_to_plot) - 1:
            stepcount_axis.set_xlabel("Episode step count", fontweight="bold", **axis_label_kwargs)
        else:
            stepcount_axis.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    fig.subplots_adjust(hspace=cfg.h_space, wspace=cfg.w_space)
    png_path = output_dir / f"{save_prefix}_episode_length_analysis.png"
    pdf_path = output_dir / f"{save_prefix}_episode_length_analysis.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
