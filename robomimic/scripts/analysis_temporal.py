from dataclasses import dataclass, field
from enum import Enum
from glob import glob
from pathlib import Path
import os
import textwrap

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
    success_rate_ci95: float | None = None
    W_episode: float | None = None
    W_total: float | None = None


@dataclass
class AnalysisConfig:
    task_names: list[str] = field(default_factory=lambda: ["lift", "can", "square"])
    dataset_split: str = "ph"
    control_frequency_hz: float = 20.0
    rollout_sampling_mode: str = "all" # "sample", "firstN", or "all"
    rollout_selection_mode: str = "combine" # "combine" or "best"
    rollout_sampling_seed: int = 2
    rollout_success_only: bool = False
    histogram_bins: int = 81
    include_rankings: bool = False
    include_demo_dataset: bool = False
    generate_tex_tables: bool = True

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
DATASET_ROOT = str(Path(__file__).parent.parent.parent.absolute()) + "/"
def default_robomimic_datasets(task_name: str, dataset_split: str) -> list[DatasetLocator]:
    return [
        DatasetLocator(path=DATASET_ROOT+f"datasets/{task_name}/{dataset_split}/image_v15.hdf5", label="Demo", dataset_type=DatasetType.DEMO),
        DatasetLocator(path=DATASET_ROOT+f"rollouts/{task_name}/{dataset_split}/bc/*.hdf5", label="BC", dataset_type=DatasetType.BASELINE),
        DatasetLocator(path=DATASET_ROOT+f"rollouts/{task_name}/{dataset_split}/bc_rnn/*.hdf5", label="BC-RNN", dataset_type=DatasetType.BASELINE),
        DatasetLocator(path=DATASET_ROOT+f"rollouts/{task_name}/{dataset_split}/diffusion_policy_mod/*.hdf5", label="DP", dataset_type=DatasetType.BASELINE),
        DatasetLocator(path=DATASET_ROOT+f"rollouts/{task_name}/{dataset_split}/tc_diffusion_policy_mod/*.hdf5", label="TC-DP", dataset_type=DatasetType.INFERENCE),
        # DatasetLocator(path=DATASET_ROOT+f"rollouts/{task_name}/{dataset_split}/diffusion_policy/*.hdf5", label="DP (robomimic default)", dataset_type=DatasetType.BASELINE),
        # DatasetLocator(path=DATASET_ROOT+f"rollouts/{task_name}/{dataset_split}/tc_diffusion_policy/*.hdf5", label="TC-DP (robomimic default)", dataset_type=DatasetType.INFERENCE),
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


def create_tex_table_from_results(
    results: dict[str, ParsedDatasetInfo],
    env_order: list[str],
    dataset_split: str,
    include_success: bool = False,
):
    task_titles = {
        "can": "Can",
        "square": "Square",
        "lift": "Lift",
        "tool_hang": "Tool Hang",
        "transport": "Transport",
    }
    metric_headers = {
        "W_episode": r"$W_{\text{episode}}$",
        "success_rate": r"Success Rate (\%)",
    }
    metrics = ["W_episode"] + (["success_rate"] if include_success else [])
    datasets = [r for r in results.values() if not r.is_demo_dataset]
    if not datasets:
        raise ValueError("No non-demo datasets available for TeX table generation.")

    method_order = {
        "BC": -2,
        "BC-RNN": -1,
        "DP": 0,
        "TC-DP": 1,
        "TC-DP (ID+SiP)": 1,
    }
    method_display_names = {
        "TC-DP": "TC-DP (ID+SiP)",
    }
    groups: dict[str, dict[str, ParsedDatasetInfo]] = {}
    for dataset in datasets:
        method = method_display_names.get(dataset.short_label or "", dataset.short_label or dataset.path)
        groups.setdefault(method, {})[dataset.task_name] = dataset

    methods = sorted(groups.keys(), key=lambda method: (method_order.get(method, 99), method))
    mins = {
        env: min(
            (float(d.W_episode) for d in datasets if d.task_name == env and d.W_episode is not None),
            default=float("inf"),
        )
        for env in env_order
    }

    def fmt_cell(env: str, d: ParsedDatasetInfo | None, metric: str):
        if d is None:
            return "x"

        value = getattr(d, metric, None)
        if value is None:
            return "x"

        if metric == "success_rate":
            ci95 = d.success_rate_ci95
            if ci95 is None:
                return rf"\textcolor{{lightgray}}{{{float(value):.1f}\%}}"
            return rf"\textcolor{{lightgray}}{{${float(value):.1f} \pm {float(ci95):.1f}\%$}}"

        text = f"{float(value):.2f}"
        if abs(float(value) - mins[env]) < 5e-3:
            return rf"\textbf{{{text}}}"
        return text

    header = " & ".join(metric_headers[m] for m in metrics)
    colspec = "l" + "c" * (len(metrics) * len(env_order))
    env_headers = " & ".join(
        rf"\multicolumn{{{len(metrics)}}}{{c}}{{\textbf{{{task_titles.get(env, env)} ({dataset_split.upper()})}}}}"
        for env in env_order
    )
    cmidrules = " ".join(
        rf"\cmidrule(lr){{{2 + i * len(metrics)}-{1 + (i + 1) * len(metrics)}}}"
        for i in range(len(env_order))
    )
    metric_row = " & ".join([header] * len(env_order))
    print(textwrap.dedent(f"""
    \\begin{{tabular}}{{{colspec}}}
    \\toprule
    \\textbf{{Method}} & {env_headers} \\\\
    {cmidrules}
    & {metric_row} \\\\
    \\midrule
    """).strip())
    for method in methods:
        row = [method]
        for env in env_order:
            dataset = groups[method].get(env)
            row.extend(fmt_cell(env, dataset, metric) for metric in metrics)
        print(" & ".join(row) + r" \\")
    print(textwrap.dedent(r"""
    \bottomrule
    \end{tabular}
    """).strip())


def extract_episode_metrics(
    dataset_path: str,
    num_episodes: int | None = None,
    horizon: int | None = None,
    success_horizon: int | None = None,
    sampling_mode: str = "firstN",
    rng: np.random.Generator | None = None,
    success_only: bool = False,
) -> tuple[list[int], list[float]]:
    with h5py.File(dataset_path, "r") as f:
        demos = sorted(list(f["data"].keys()), key=lambda name: int(name.split("_")[-1]))
        episode_infos = []
        for demo in demos:
            demo_group = f["data"][demo]
            length = demo_group.attrs.get("num_samples", None)
            if length is None:
                length = demo_group["actions"].shape[0]
            raw_length = int(length)

            is_success = False
            if "rewards" in demo_group:
                reward_values = np.asarray(demo_group["rewards"][:], dtype=float)
                if success_horizon is not None:
                    reward_values = reward_values[:success_horizon]
                is_success = bool(np.max(reward_values) == 1.0) if reward_values.size > 0 else False

            episode_infos.append((raw_length, float(is_success)))

        if success_only:
            episode_infos = [info for info in episode_infos if info[1] > 0.5]

        if sampling_mode == "all":
            num_episodes = None

        if num_episodes is not None:
            num_episodes = min(num_episodes, len(episode_infos))
            if sampling_mode == "sample":
                if rng is None:
                    raise ValueError("rng must be provided when sampling_mode='sample'.")
                sampled_indices = np.sort(rng.choice(len(episode_infos), size=num_episodes, replace=False))
                episode_infos = [episode_infos[idx] for idx in sampled_indices]
            elif sampling_mode == "firstN":
                episode_infos = episode_infos[:num_episodes]
            else:
                raise ValueError(f"Unsupported sampling_mode: {sampling_mode!r}")

        lengths = []
        successes = []
        for raw_length, is_success in episode_infos:
            if horizon is not None:
                length = min(raw_length, horizon)
            else:
                length = raw_length
            lengths.append(int(length))
            successes.append(float(is_success))
    return lengths, successes


def count_dataset_episodes(dataset_path: str) -> int:
    with h5py.File(dataset_path, "r") as f:
        return len(f["data"])


def resolve_horizon(task_name: str, dataset_split: str) -> int:
    if dataset_split == "ph" and task_name in {"can", "square", "lift"}:
        return 400
    if task_name == "transport":
        return 1100 if dataset_split == "mh" else 700
    return 500


def task_name_from_path(path: str) -> str:
    normalized = Path(path)
    for part in normalized.parts:
        if part in TASK_NAMES:
            return part

    raise ValueError(f"Could not infer task name from {path}")


def success_rate_percent(success_flags: list[float]) -> float | None:
    if not success_flags:
        return None
    return 100.0 * float(np.mean(np.asarray(success_flags) > 0.5))


@draccus.wrap()
def main(cfg: AnalysisConfig):
    invalid_task_names = [task_name for task_name in cfg.task_names if task_name not in TASK_NAMES]
    if invalid_task_names:
        raise ValueError(
            f"task_names must be drawn from {sorted(TASK_NAMES)}, got invalid entries {invalid_task_names!r}"
        )
    if cfg.dataset_split not in {"mh", "ph"}:
        raise ValueError("dataset_split must be 'mh' or 'ph', got {!r}".format(cfg.dataset_split))
    if cfg.rollout_sampling_mode not in {"firstN", "sample", "all"}:
        raise ValueError(
            "rollout_sampling_mode must be 'firstN', 'sample', or 'all', got {!r}".format(cfg.rollout_sampling_mode)
        )
    if cfg.rollout_selection_mode not in {"combine", "best"}:
        raise ValueError(
            "rollout_selection_mode must be 'combine' or 'best', got {!r}".format(cfg.rollout_selection_mode)
        )
    all_parsed_datasets = {}
    dataset_types = {DatasetType.DEMO, DatasetType.BASELINE, DatasetType.INFERENCE}

    for task_name in cfg.task_names:
        datasets = default_robomimic_datasets(task_name, cfg.dataset_split)
        parsed_datasets = parse_datasets(cfg, task_name, select_datasets_by_type(datasets, dataset_types))
        all_parsed_datasets.update(parsed_datasets)
        generate_graphs(
            cfg,
            task_name,
            parsed_datasets,
            save_prefix=f"robomimic_{task_name}_{cfg.dataset_split}",
        )

    if cfg.generate_tex_tables:
        create_tex_table_from_results(
            all_parsed_datasets,
            env_order=list(cfg.task_names),
            dataset_split=cfg.dataset_split,
            include_success=True,
        )


def parse_datasets(cfg: AnalysisConfig, task_name: str, datasets: list[DatasetLocator]) -> dict[str, ParsedDatasetInfo]:
    from scipy.stats import wasserstein_distance

    parsed_datasets = {}
    horizon = resolve_horizon(task_name, cfg.dataset_split)
    rollout_rng = np.random.default_rng(cfg.rollout_sampling_seed)
    default_task_name = task_name
    demo_step_counts: list[int] | None = None

    demo_dataset = next((d for d in datasets if d.dataset_type == DatasetType.DEMO and os.path.exists(d.path)), None)
    if demo_dataset is None:
        raise ValueError("Could not resolve rollout episode budget because no demo dataset was found.")
    rollout_episode_budget = count_dataset_episodes(demo_dataset.path)
    print(f"Rollout episode budget derived from demo dataset: {rollout_episode_budget}")

    for dataset in datasets:
        if dataset.dataset_type == DatasetType.DEMO:
            if not os.path.exists(dataset.path):
                print(f"Skipping missing demo dataset: {dataset.path}")
                continue

            print(f"Parsing dataset: {dataset.label or 'Demo'}")
            step_counts, success_flags = extract_episode_metrics(
                dataset.path,
                None,
                horizon=horizon,
                success_horizon=horizon,
            )
            demo_step_counts = step_counts
            success_rate = success_rate_percent(success_flags)
            dataset_task_name = task_name_from_path(dataset.path)
            parsed_datasets[dataset.path] = ParsedDatasetInfo(
                path=dataset.path,
                short_label=dataset.label or "Demo",
                is_demo_dataset=True,
                task_name=dataset_task_name,
                dataset_type=dataset.dataset_type,
                step_counts=step_counts,
                success_rate=success_rate if success_rate is not None else 100.0,
                success_rate_ci95=0.0,
            )
            print("Demo episodes: {} ({})".format(len(step_counts), dataset.label or "Demo"))
            continue

        rollout_paths = sorted(glob(dataset.path))
        if not rollout_paths:
            print(f"Skipping missing rollout datasets: {dataset.path}")
            continue
        if demo_step_counts is None:
            raise ValueError(f"Demo dataset must be parsed before rollout datasets for task {task_name}.")

        rollout_candidates = []
        base_episodes_per_rollout, extra_episodes = divmod(rollout_episode_budget, len(rollout_paths))
        demo_step_counts_np = np.asarray(demo_step_counts, dtype=float)

        for rollout_idx, rollout_path in enumerate(rollout_paths):
            rollout_num_episodes = base_episodes_per_rollout + int(rollout_idx < extra_episodes)
            step_counts, success_flags = extract_episode_metrics(
                rollout_path,
                rollout_num_episodes,
                horizon=horizon,
                success_horizon=horizon,
                sampling_mode=cfg.rollout_sampling_mode,
                rng=rollout_rng,
                success_only=cfg.rollout_success_only,
            )
            if not step_counts:
                print(f"Skipping rollout dataset without episodes: {rollout_path}")
                continue

            rollout_task_name = task_name_from_path(rollout_path)
            success_rate = success_rate_percent(success_flags)
            wd = wasserstein_distance(np.asarray(step_counts, dtype=float), demo_step_counts_np) / cfg.control_frequency_hz
            rollout_candidates.append({
                "path": rollout_path,
                "task_name": rollout_task_name,
                "step_counts": step_counts,
                "success_rate": success_rate,
                "wd": float(wd),
            })
            print(
                f"Parsing rollout dataset: {dataset.label or 'Rollout'} "
                f"({rollout_path}) with {len(step_counts)} episodes, W_episode={wd:.4f}"
            )

        if not rollout_candidates:
            print(f"Skipping rollout dataset without episodes: {dataset.path}")
            continue

        label = dataset.label or Path(rollout_paths[0]).parent.name
        print(f"Rollout episode budget for {default_task_name}: {rollout_episode_budget}")

        if cfg.rollout_selection_mode == "best":
            selected = min(
                rollout_candidates,
                key=lambda item: (item["wd"], -(item["success_rate"] or float("-inf")), item["path"]),
            )
            parsed_datasets[dataset.path] = ParsedDatasetInfo(
                path=dataset.path,
                short_label=label,
                is_demo_dataset=False,
                task_name=selected["task_name"] or default_task_name,
                dataset_type=dataset.dataset_type or DatasetType.UNKNOWN,
                step_counts=selected["step_counts"],
                success_rate=selected["success_rate"],
                success_rate_ci95=0.0 if selected["success_rate"] is not None else None,
                W_episode=selected["wd"],
                W_total=selected["wd"],
            )
            print(
                f"Selected best rollout for {label}: {selected['path']} "
                f"(W_episode={selected['wd']:.4f}, episodes={len(selected['step_counts'])})"
            )
            continue

        combined_step_counts = []
        rollout_success_rates = []
        combined_task_name = None
        for candidate in rollout_candidates:
            if combined_task_name is None:
                combined_task_name = candidate["task_name"]
            combined_step_counts.extend(candidate["step_counts"])
            if candidate["success_rate"] is not None:
                rollout_success_rates.append(candidate["success_rate"])

        success_rate_mean = float(np.mean(rollout_success_rates)) if rollout_success_rates else None
        success_rate_ci95 = None
        if len(rollout_success_rates) >= 2:
            success_rate_ci95 = 1.96 * float(np.std(rollout_success_rates, ddof=1)) / np.sqrt(len(rollout_success_rates))
        elif len(rollout_success_rates) == 1:
            success_rate_ci95 = 0.0
        parsed_datasets[dataset.path] = ParsedDatasetInfo(
            path=dataset.path,
            short_label=label,
            is_demo_dataset=False,
            task_name=combined_task_name or default_task_name,
            dataset_type=dataset.dataset_type or DatasetType.UNKNOWN,
            step_counts=combined_step_counts,
            success_rate=success_rate_mean,
            success_rate_ci95=success_rate_ci95,
        )
        print("Combined episodes: {} ({})".format(len(combined_step_counts), label))

    calculate_wasserstein_distances(parsed_datasets, control_frequency_hz=cfg.control_frequency_hz)
    return parsed_datasets


def calculate_wasserstein_distances(
    parsed_datasets: dict[str, ParsedDatasetInfo],
    control_frequency_hz: float = 20.0,
):
    from scipy.stats import wasserstein_distance

    if control_frequency_hz <= 0:
        raise ValueError(f"control_frequency_hz must be > 0, got {control_frequency_hz}")

    demonstration_dataset = next(d for d in parsed_datasets.values() if d.is_demo_dataset)
    demo_steps = np.asarray(demonstration_dataset.step_counts, dtype=float)

    for dataset_path, parsed_info in parsed_datasets.items():
        step_counts = np.asarray(parsed_info.step_counts, dtype=float)
        w_distance_step = wasserstein_distance(step_counts, demo_steps) / control_frequency_hz
        parsed_datasets[dataset_path].W_episode = w_distance_step
        parsed_datasets[dataset_path].W_total = w_distance_step


def generate_graphs(cfg: AnalysisConfig, task_name: str, parsed_datasets: dict[str, ParsedDatasetInfo], save_prefix: str):
    output_dir = Path(__file__).resolve().parent / "analysis_temporal"
    output_dir.mkdir(parents=True, exist_ok=True)

    demonstration_dataset = next(d for d in parsed_datasets.values() if d.is_demo_dataset)
    horizon = resolve_horizon(task_name, cfg.dataset_split)
    bins_step_counts = np.linspace(0.0, float(horizon), cfg.histogram_bins + 1)
    ref_step_hist, _ = np.histogram(demonstration_dataset.step_counts, bins=bins_step_counts)
    step_centers = 0.5 * (bins_step_counts[:-1] + bins_step_counts[1:])
    step_widths = np.diff(bins_step_counts)

    rows_to_plot = [
        (path, info)
        for path, info in parsed_datasets.items()
        if cfg.include_demo_dataset or not info.is_demo_dataset
    ]
    if not rows_to_plot:
        rows_to_plot = [
            (path, info)
            for path, info in parsed_datasets.items()
            if info.is_demo_dataset
        ]
        if not rows_to_plot:
            raise ValueError("No datasets to plot.")

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
        stepcount_axis.set_xlim(0, horizon)

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
