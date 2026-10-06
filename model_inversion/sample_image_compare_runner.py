# Generated with Claude Code

"""Combine the "sample image logging (v2)" percentile comparison data saved by
model_inversion_evaluate.py across multiple PPA-main wandb runs into per-class
side-by-side comparison figures (two columns -- eval confidence, target confidence --
per selected run, one figure per class), and log the results to a new wandb run.
"""

import os
import time

import matplotlib.pyplot as plt
import torch
import wandb

import sys
sys.path.append(os.getcwd())

from pick import pick

from model_inversion.model_inversion_evaluate import render_percentile_column, render_reference_row
from model_inversion.run_wandb_evaluate import pick_runs_to_evaluate
from data_processing.datasets import load_example_images

ENTITY = "BreuerLab"
SOURCE_PROJECT = "PPA-evaluation"
DEST_PROJECT = "PPA-evaluation"

# Number of real reference images shown per class in the top row of comparison figures.
NUM_REFERENCE_IMAGES = 3


def discover_classes(run):
    """List the class labels a run has saved sample-image-v2 data for."""
    prefix = f"model_inversion/results/sample_image_v2/{run.id}/class_"
    suffix = "/data.pt"
    classes = []
    for f in run.files():
        if f.name.startswith(prefix) and f.name.endswith(suffix):
            try:
                classes.append(int(f.name[len(prefix):-len(suffix)]))
            except ValueError:
                continue
    return classes


def get_or_prompt_header(run):
    """Return a short display header for `run` to use in combined sample-image figures.

    Checks the run's own wandb config for a previously-stored header first; only
    prompts (and persists the answer back onto the run) if none is stored yet.
    """
    header = run.config.get('sample_image_v2_header', None)
    if header:
        print(f"Using stored header for {run.name} ({run.id}): {header}")
        return header

    header = input(f"Header for run {run.name} ({run.id}) in combined sample images: ").strip()
    run.config['sample_image_v2_header'] = header
    run.update()
    return header


def download_columns(run, class_label):
    """Download and load a run's saved sample-image-v2 data for one class. Returns
    None if missing."""
    remote_path = f"model_inversion/results/sample_image_v2/{run.id}/class_{class_label}/data.pt"
    try:
        downloaded = run.file(remote_path).download(exist_ok=True, replace=True)
    except Exception:
        print(f"Run {run.name} ({run.id}) has no sample image logging (v2) data "
              f"for class {class_label}, skipping.")
        return None
    return torch.load(downloaded.name)


def build_comparison_figure(run_columns, run_headers, class_label=None):
    """run_columns: list of (run, columns_dict) pairs, all for the same class and
    all with the same num_imgs_per_col. run_headers: {run.id: short display header}.

    Row 0 shows NUM_REFERENCE_IMAGES real example images of `class_label` (looked up via
    the first run's saved 'dataset' config field -- all compared runs are assumed to share
    the same dataset), centered as a group across each run's pair of columns. The run's
    header is repeated on both columns, just above the '(eval conf)'/'(target conf)' labels.
    """
    num_imgs_per_col = min(columns['num_imgs_per_col'] for _, columns in run_columns)
    num_runs = len(run_columns)
    nrows = num_imgs_per_col + 1

    dataset_name = run_columns[0][0].config.get('dataset', None)
    ref_imgs = load_example_images(dataset_name, class_label, num_images=NUM_REFERENCE_IMAGES) \
        if dataset_name else []

    fig = plt.figure(figsize=(3 * num_runs, 2.2 * nrows))
    gs = fig.add_gridspec(nrows, 2 * num_runs)
    ref_gs = gs[0, :].subgridspec(1, NUM_REFERENCE_IMAGES)
    render_reference_row(
        [fig.add_subplot(ref_gs[0, i]) for i in range(NUM_REFERENCE_IMAGES)], ref_imgs)

    for i, (run, columns) in enumerate(run_columns):
        percentiles = columns['percentiles'][:num_imgs_per_col]
        header = run_headers[run.id]

        render_percentile_column(
            [fig.add_subplot(gs[row, 2 * i]) for row in range(1, nrows)],
            columns['left'], percentiles, header=f'{header}\n(eval conf)')
        render_percentile_column(
            [fig.add_subplot(gs[row, 2 * i + 1]) for row in range(1, nrows)],
            columns['right'], percentiles, header=f'{header}\n(target conf)')

    if class_label is not None:
        fig.suptitle(f'Class {class_label}')
    fig.tight_layout()
    return fig


def pick_classes_to_highlight(all_classes):
    """Prompt the user to pick a subset of `all_classes` to combine side by side into
    a single figure, instead of one figure per class. Returns None if the user opts
    out (keep the default one-figure-per-class behavior)."""
    title = "Combine classes into one side-by-side figure instead of one figure per class?"
    choice, _ = pick(["No, one figure per class", "Yes, pick classes to combine"], title)
    if choice.startswith("No"):
        return None

    title = "Choose classes to highlight (space to toggle, enter to confirm):"
    options = [str(c) for c in all_classes]
    selected = pick(options, title, multiselect=True, min_selection_count=1)
    return [int(label) for label, _ in selected]


def build_multi_class_comparison_figure(class_run_columns, run_headers):
    """class_run_columns: {class_label: run_columns} where each run_columns is a list of
    (run, columns_dict) pairs, as in build_comparison_figure. run_headers: {run.id: short
    display header}.

    Builds a single figure with each class's set of run columns placed adjacent to each
    other (left-to-right, one class's block after another), with NUM_REFERENCE_IMAGES
    ground-truth reference images centered as a group above just that class's own block
    of columns.
    """
    num_imgs_per_col = min(
        columns['num_imgs_per_col']
        for run_columns in class_run_columns.values()
        for _, columns in run_columns)
    nrows = num_imgs_per_col + 1

    cols_per_class = {cls: 2 * len(run_columns) for cls, run_columns in class_run_columns.items()}
    total_cols = sum(cols_per_class.values())

    fig = plt.figure(figsize=(1.5 * total_cols, 2.2 * nrows))
    gs = fig.add_gridspec(nrows, total_cols)

    col_offset = 0
    for class_label, run_columns in class_run_columns.items():
        n_cols = cols_per_class[class_label]

        dataset_name = run_columns[0][0].config.get('dataset', None)
        ref_imgs = load_example_images(dataset_name, class_label, num_images=NUM_REFERENCE_IMAGES) \
            if dataset_name else []
        ref_gs = gs[0, col_offset:col_offset + n_cols].subgridspec(1, NUM_REFERENCE_IMAGES)
        render_reference_row(
            [fig.add_subplot(ref_gs[0, i]) for i in range(NUM_REFERENCE_IMAGES)],
            ref_imgs, label=f'Class {class_label}')

        for i, (run, columns) in enumerate(run_columns):
            percentiles = columns['percentiles'][:num_imgs_per_col]
            header = run_headers[run.id]
            col = col_offset + 2 * i

            render_percentile_column(
                [fig.add_subplot(gs[row, col]) for row in range(1, nrows)],
                columns['left'], percentiles, header=f'{header}\n(eval conf)')
            render_percentile_column(
                [fig.add_subplot(gs[row, col + 1]) for row in range(1, nrows)],
                columns['right'], percentiles, header=f'{header}\n(target conf)')

        col_offset += n_cols

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    selected_runs = pick_runs_to_evaluate(ENTITY, SOURCE_PROJECT)
    if not selected_runs:
        raise RuntimeError("No runs selected.")

    run_headers = {run.id: get_or_prompt_header(run) for run in selected_runs}

    classes_by_run = {run.id: discover_classes(run) for run in selected_runs}
    all_classes = sorted({cls for classes in classes_by_run.values() for cls in classes})
    if not all_classes:
        raise RuntimeError("None of the selected runs had sample image logging (v2) data.")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    comparison_run = wandb.init(
        entity=ENTITY,
        project=DEST_PROJECT,
        name=f"sample_image_v2_comparison_{timestamp}",
    )
    comparison_run.config.update({'compared_runs': [run.id for run in selected_runs]})

    def collect_run_columns(class_label):
        run_columns = []
        for run in selected_runs:
            if class_label not in classes_by_run[run.id]:
                print(f"Run {run.name} ({run.id}) has no data for class {class_label}, "
                      f"skipping it for this class.")
                continue
            columns = download_columns(run, class_label)
            if columns is not None:
                run_columns.append((run, columns))
        return run_columns

    highlighted_classes = pick_classes_to_highlight(all_classes)

    if highlighted_classes is not None:
        class_run_columns = {}
        for class_label in highlighted_classes:
            run_columns = collect_run_columns(class_label)
            if run_columns:
                class_run_columns[class_label] = run_columns

        if not class_run_columns:
            raise RuntimeError("None of the selected classes had data for the selected runs.")

        fig = build_multi_class_comparison_figure(class_run_columns, run_headers)

        classes_str = "_".join(str(c) for c in class_run_columns)
        out_dir = "model_inversion/results/sample_image_v2_comparisons/combined"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"comparison_{timestamp}_classes_{classes_str}.png")
        fig.savefig(out_path)
        print(f"Saved combined classes {classes_str} comparison figure to {out_path}")

        wandb.log({f'Sample Image Logging (v2) Comparison/combined_classes_{classes_str}': wandb.Image(fig)})
        wandb.save(out_path)
        plt.close(fig)
    else:
        for class_label in all_classes:
            run_columns = collect_run_columns(class_label)
            if not run_columns:
                continue

            fig = build_comparison_figure(run_columns, run_headers, class_label=class_label)

            out_dir = f"model_inversion/results/sample_image_v2_comparisons/class_{class_label}"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"comparison_{timestamp}.png")
            fig.savefig(out_path)
            print(f"Saved class {class_label} comparison figure to {out_path}")

            wandb.log({f'Sample Image Logging (v2) Comparison/class_{class_label}': wandb.Image(fig)})
            wandb.save(out_path)
            plt.close(fig)

    wandb.finish()
