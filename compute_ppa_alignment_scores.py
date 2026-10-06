# Generated with Claude Code
"""Post-hoc gradient-manifold alignment score computation for a completed PPA run.

Downloads the raw (w, grad_x) log artifact saved unconditionally during the attack,
reconstructs the generator and attack transformations from the run's W&B config, then
computes alignment scores (Peng et al., NeurIPS'25) offline and uploads the result CSV
back to the original run.

Usage:
    python compute_ppa_alignment_scores.py
    python compute_ppa_alignment_scores.py --chunk_size 40 --max_samples 4
"""

import argparse
import ast
import csv
import inspect
import json
import os
import sys
import time
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb

from omegaconf import OmegaConf
from pick import pick

from Plug_and_Play_Attacks.utils.attack_config_parser import AttackConfigParser
from Plug_and_Play_Attacks.utils.stylegan import load_generator
import model_inversion.plug_and_play.modify_to_ppa_repo as ppa_modify
from model_inversion.metrics.mi_gradient_alignment import (
    compute_gradient_alignment_score,
    compute_alignment_scores_from_raw_log,
    make_det_transformations,
)
from utils.wandb_helpers import get_config
from utils.load_trained_models import get_target_config_and_weights
from experiments.all_run_ids import ALL_RUN_IDS

ENTITY = "BreuerLab"
DEFAULT_PROJECT = "PPA-main"
ALIGN_PROJECT = "align_scores"
MIN_RUN_DATE = "2026-05-01"  # only offer runs created on/after this date (ISO string compare)


def _has_alignment_raw_log(run):
    raw_log_path = f"model_inversion/plug_and_play/results/alignment_raw/{run.id}.pt"
    try:
        _ = run.file(raw_log_path).size
        return True
    except Exception:
        return False


def pick_target_models():
    """Stage-1 picker mirroring run_wandb_evaluate.py: pick target model group(s) from
    ALL_RUN_IDS, then individual model(s). Returns {model_name: target_wandb_id}."""
    selected_groups = pick(
        list(ALL_RUN_IDS.keys()),
        "Select run ID group(s):",
        multiselect=True, min_selection_count=1,
    )
    pooled = {}
    for name, _ in selected_groups:
        pooled.update(ALL_RUN_IDS[name])

    options = list(pooled.keys()) + ["All models"]
    selected_models = pick(
        options,
        "Select target model(s):",
        multiselect=True, min_selection_count=1,
    )
    selected_names = [name for name, _ in selected_models]
    return pooled if "All models" in selected_names else {n: pooled[n] for n in selected_names}


VIZ_TARGETS = [1, 6, 7, 486, 12, 16, 500, 503, 506, 509]  # FaceScrub viz-mode target classes
TARGETS_100_MIN_DATE = "2026-08-01"  # targets=100 runs only qualify if created on/after this date
VIZ_ADAM_LR = 0.005  # required attack.attack.optimizer.Adam.lr
RUN_FILTER_DESC = (f"Adam lr={VIZ_ADAM_LR} and (targets containing {VIZ_TARGETS}, "
                   f"or targets=100 created since {TARGETS_100_MIN_DATE})")


def _is_grad_viz_run(run):
    """Whether the run's attack.attack.optimizer.Adam.lr is VIZ_ADAM_LR and its
    attack.attack.targets is a list containing every class in VIZ_TARGETS, or is 100 with the
    run created on/after TARGETS_100_MIN_DATE."""
    attack_cfg = run.config.get('attack', {}).get('attack', {})
    lr = attack_cfg.get('optimizer', {}).get('Adam', {}).get('lr')
    if lr is None or not np.isclose(lr, VIZ_ADAM_LR):
        return False
    targets = attack_cfg.get('targets')
    if isinstance(targets, list):
        return set(VIZ_TARGETS) <= set(targets)
    return targets == 100 and str(run.created_at) >= TARGETS_100_MIN_DATE


def pick_runs_to_score(project=DEFAULT_PROJECT, csv_status=False, det_transforms=True, seed=None, priors_out=None,
                       require_viz_targets=False):
    """Two-stage picker mirroring run_wandb_evaluate.py:
      1. Pick target model group(s) from ALL_RUN_IDS, then individual model(s) (see pick_target_models).
      2. For each selected model, list PPA-main runs (created on/after MIN_RUN_DATE) targeting
         it that have the alignment raw log saved; show a per-model picker (none = skip that model).
         If csv_status is set, the picker is split into runs that already have a post-hoc
         alignment CSV on W&B (with the number of classes it covers) and runs that don't.
      3. If priors_out (a dict) is given, right after each model's runs are chosen, prompt for
         each chosen run's optional prior run (see _pick_prior_run) and store
         priors_out[run.id] = (prior_run, prior_df).
    If require_viz_targets is set, only runs passing _is_grad_viz_run are offered.
    Returns a list of wandb Run objects.
    """
    model_to_id = pick_target_models()

    api = wandb.Api()
    runs_to_score = []
    for model_name, target_wandb_id in model_to_id.items():
        print(f"Scanning {project} for runs targeting {model_name}...")
        candidates = api.runs(
            f"{ENTITY}/{project}",
            filters={"config.target_wandb_id": target_wandb_id},
        )
        matches = [run for run in candidates
                   if str(run.created_at) >= MIN_RUN_DATE
                   and (not require_viz_targets or _is_grad_viz_run(run))
                   and _has_alignment_raw_log(run)]
        if not matches:
            viz_note = f" and {RUN_FILTER_DESC}" if require_viz_targets else ""
            print(f"  No runs since {MIN_RUN_DATE} with alignment raw log{viz_note} found for {model_name}, skipping.")
            continue

        if csv_status:
            with_csv, without_csv = [], []
            for run in matches:
                num_classes = _fetch_num_csv_classes(run)
                (with_csv if num_classes is not None else without_csv).append((run, num_classes))
            with_csv.sort(key=lambda x: x[0].created_at)
            without_csv.sort(key=lambda x: x[0].created_at)
            matches = [run for run, _ in with_csv] + [run for run, _ in without_csv]
            options = (
                [f"[HAS CSV: {num_classes} classes]  {run.name}  ({run.id})  [{run.state}]" for run, num_classes in with_csv]
                + [f"[NO CSV]  {run.name}  ({run.id})  [{run.state}]" for run, _ in without_csv]
            )
        else:
            options = [f"{run.name}  ({run.id})  [{run.state}]" for run in matches]
        options.append("All runs")
        selected = pick(
            options,
            f"{model_name}: choose run(s) to score (none = skip):",
            multiselect=True, min_selection_count=0,
        )
        selected_idxs = [i for _, i in selected]
        chosen = list(matches) if len(matches) in selected_idxs else [matches[i] for i in selected_idxs]
        if priors_out is not None:
            for r in chosen:
                priors_out[r.id] = _pick_prior_run(r, project, det_transforms, seed)
        runs_to_score += chosen

    return runs_to_score


GRAD_PLOT_PRESETS_FILE = os.path.join("experiments", "grad_plot_presets.csv")


def _load_grad_plot_presets(project):
    """Saved gradient-plot run selections for `project`. Returns {name: [run_id, ...]}."""
    if not os.path.exists(GRAD_PLOT_PRESETS_FILE):
        return {}
    with open(GRAD_PLOT_PRESETS_FILE, newline="") as f:
        return {row["name"]: json.loads(row["run_ids"])
                for row in csv.DictReader(f) if row["project"] == project}


def _save_grad_plot_preset(name, project, run_ids):
    """Append a named run selection to the gradient-plot preset file, creating it if needed."""
    file_exists = os.path.exists(GRAD_PLOT_PRESETS_FILE)
    os.makedirs(os.path.dirname(GRAD_PLOT_PRESETS_FILE), exist_ok=True)
    with open(GRAD_PLOT_PRESETS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "project", "run_ids"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({"name": name, "project": project, "run_ids": json.dumps(run_ids)})


def pick_runs_with_preset(project=DEFAULT_PROJECT):
    """Offer a saved run selection first; otherwise fall back to pick_runs_to_score and offer
    to save the chosen runs as a preset so they don't need re-selecting next time."""
    presets = _load_grad_plot_presets(project)
    if presets:
        none_option = "None (select runs manually)"
        options = [f"{name}  ({len(run_ids)} runs)" for name, run_ids in presets.items()] + [none_option]
        _, idx = pick(options, "Use a saved gradient-plot run preset?")
        if idx < len(presets):
            name, run_ids = list(presets.items())[idx]
            api = wandb.Api()
            runs = [api.run(f"{ENTITY}/{project}/{run_id}") for run_id in run_ids]
            print(f"Using preset '{name}': {[f'{r.name} ({r.id})' for r in runs]}")
            return runs

    runs = pick_runs_to_score(project=project, require_viz_targets=True)
    if runs and input("Save these runs as a preset? (y/n): ").strip().lower().startswith("y"):
        name = input("Enter a name for this preset: ").strip()
        _save_grad_plot_preset(name, project, [r.id for r in runs])
        print(f"Saved preset '{name}' to {GRAD_PLOT_PRESETS_FILE}.")
    return runs


def _has_align15(run):
    """Whether the run's cached 'align15' summary is True (bool, or the string "True")."""
    return run.summary.get('align15') in (True, "True")


def _set_align15_summary(run, num_classes):
    """Record on the run's W&B summary whether it has a full 15-class alignment CSV."""
    has_15 = num_classes is not None and num_classes >= 15
    run.summary['align15'] = has_15
    run.summary.update()
    return has_15


def _get_align15_mean_score(run):
    """Best-effort mean alignment score for a run known to have align15 coverage: prefers the
    cached 'avg_alignment_score_posthoc' summary value (no download needed); only falls back to
    reading its alignment CSV directly if that summary field wasn't logged."""
    mean_score = run.summary.get('avg_alignment_score_posthoc')
    if mean_score is not None:
        return mean_score
    csv_path = _fetch_recalc_csv(run)
    if csv_path is None:
        return None
    return float(pd.read_csv(csv_path)['alignment_score'].mean())


DEFAULT_ALIGN15_CSV_OUT = "scripts/csvs/target_align15_scores.csv"


def list_runs_missing_full_classes(project=DEFAULT_PROJECT, required_classes=15,
                                    align15_csv_out=DEFAULT_ALIGN15_CSV_OUT):
    """For user-selected target model(s), find those where NO run (created on/after MIN_RUN_DATE)
    has an alignment CSV covering `required_classes` classes, and print them all together at the
    end. A target model with at least one such run is considered covered. Also records an
    'align15' summary flag (True/False) on each checked run, and writes a CSV to
    `align15_csv_out` mapping each covered target model's target_wandb_id to the mean alignment
    score (avg_alignment_score_posthoc) of the run that covers it.

    To avoid wasting time: for a target model, if any run already has a cached align15=True
    summary, no CSV is downloaded for that model at all -- its cached score is used directly and
    every other run for that model is skipped. Only once no run has a cached align15=True does it
    fall back to checking each run's CSV one at a time, stopping as soon as one with
    `required_classes` classes is found.
    """
    model_to_id = pick_target_models()

    api = wandb.Api()
    missing_models = []  # (model_name, target_wandb_id, best_num_classes_seen or None)
    align15_scores = []  # rows for align15_csv_out
    for model_name, target_wandb_id in model_to_id.items():
        print(f"Scanning {project} for runs targeting {model_name}...")
        candidates = api.runs(
            f"{ENTITY}/{project}",
            filters={"config.target_wandb_id": target_wandb_id},
        )
        matches = [run for run in candidates
                   if str(run.created_at) >= MIN_RUN_DATE and _has_alignment_raw_log(run)]
        if not matches:
            print(f"  No runs since {MIN_RUN_DATE} with alignment raw log found for {model_name}.")
            missing_models.append((model_name, target_wandb_id, None))
            continue

        # Cheap pass first: a run whose align15 flag is already cached True needs no CSV lookup
        # at all -- use it directly and skip every other run for this model.
        covered_run = next((run for run in matches if _has_align15(run)), None)

        best = None
        if covered_run is not None:
            print(f"  [{covered_run.id}] align15 summary already True — skipping CSV lookup entirely.")
        else:
            for run in matches:
                num_classes = _fetch_num_csv_classes(run)
                _set_align15_summary(run, num_classes)
                if num_classes is not None and (best is None or num_classes > best):
                    best = num_classes
                if num_classes is not None and num_classes >= required_classes:
                    covered_run = run
                    break  # no need to check this model's remaining runs once one qualifies

        if covered_run is not None:
            print(f"  Covered: a run for {model_name} already has a {required_classes}-class alignment CSV.")
            # align15 (fixed at >=15) may differ from required_classes if the caller overrode it.
            if _has_align15(covered_run):
                mean_score = _get_align15_mean_score(covered_run)
                align15_scores.append({
                    "target_wandb_id": target_wandb_id,
                    "model_name": model_name,
                    "source_run_id": covered_run.id,
                    "avg_alignment_score_posthoc": mean_score,
                })
            else:
                print(f"  (not added to {align15_csv_out}: covers required_classes={required_classes} "
                      "but not the fixed 15-class align15 threshold)")
        else:
            missing_models.append((model_name, target_wandb_id, best))

    print(f"\nTarget models with no {required_classes}-class alignment CSV on any run "
          f"({len(missing_models)}/{len(model_to_id)}):")
    if not missing_models:
        print("  (none — every selected target model is covered)")
    else:
        for model_name, target_wandb_id, best in missing_models:
            status = f"best so far: {best} classes" if best is not None else "no runs with an alignment CSV"
            print(f"  {model_name}  (target_wandb_id={target_wandb_id}): {status}")

    if align15_scores:
        os.makedirs(os.path.dirname(align15_csv_out), exist_ok=True)
        pd.DataFrame(align15_scores).to_csv(align15_csv_out, index=False)
        print(f"\nSaved {len(align15_scores)} target -> 15-class mean alignment score row(s) "
              f"to {align15_csv_out}")
    else:
        print(f"\nNo 15-class alignment scores found — {align15_csv_out} not written.")


def combine_runs_into_third(project=DEFAULT_PROJECT, det_transforms=True, seed=None):
    """For a chosen target model, pick two PPA-main runs that already have alignment CSVs on
    W&B, concatenate them (skipping any row -- keyed by batch/iteration/target_classes -- that's
    a duplicate), and upload the combined CSV, comparison histogram, and summary metrics into a
    third chosen run of the same target model. Anything already at the destination's own CSV
    path is preserved and merged in the same way, rather than overwritten.

    Read-only for the two source runs; only the destination run's W&B page (and the local
    ALIGN_OUT_DIR) are written to.
    """
    model_to_id = pick_target_models()
    api = wandb.Api()

    for model_name, target_wandb_id in model_to_id.items():
        print(f"\nScanning {project} for runs targeting {model_name}...")
        candidates = api.runs(
            f"{ENTITY}/{project}",
            filters={"config.target_wandb_id": target_wandb_id},
        )
        matches = [run for run in candidates
                   if str(run.created_at) >= MIN_RUN_DATE and _has_alignment_raw_log(run)]
        if len(matches) < 3:
            print(f"  Need at least 3 eligible runs for {model_name} (found {len(matches)}) — skipping.")
            continue

        def _label(run):
            num_classes = _fetch_num_csv_classes(run)
            status = f"{num_classes} classes" if num_classes is not None else "NO CSV"
            return f"[{status}]  {run.name}  ({run.id})  [{run.state}]"

        options = [_label(run) for run in matches]
        _, idx1 = pick(options, f"{model_name}: choose the FIRST run to combine:")
        run1 = matches[idx1]

        remaining2 = [run for i, run in enumerate(matches) if i != idx1]
        options2 = [_label(run) for run in remaining2]
        _, idx2 = pick(options2, f"{model_name}: choose the SECOND run to combine with {run1.name}:")
        run2 = remaining2[idx2]

        remaining3 = [run for run in matches if run.id not in (run1.id, run2.id)]
        options3 = [_label(run) for run in remaining3]
        _, idx3 = pick(options3, f"{model_name}: choose the DESTINATION run to store the "
                                 "combined results in:")
        dest_run = remaining3[idx3]

        csv1_path = _fetch_recalc_csv(run1)
        csv2_path = _fetch_recalc_csv(run2)
        if csv1_path is None or csv2_path is None:
            missing_run = run1.name if csv1_path is None else run2.name
            print(f"  {missing_run} has no alignment CSV on W&B — skipping {model_name}.")
            continue
        df1 = pd.read_csv(csv1_path)
        df2 = pd.read_csv(csv2_path)

        os.makedirs(ALIGN_OUT_DIR, exist_ok=True)
        csv_path, inloop_csv_path, plot_path = _recalc_paths(dest_run.id, det_transforms, seed)
        dest_existing_csv = _fetch_recalc_csv(dest_run)
        base_df = None
        if dest_existing_csv is not None:
            csv_path = dest_existing_csv
            plot_path = csv_path[:-len(".csv")] + "_comparison.png"
            base_df = pd.read_csv(csv_path)

        merged_df, added1 = _merge_new_rows(base_df, df1)
        print(f"  Added {added1}/{len(df1)} rows from {run1.name} ({run1.id}).")
        merged_df, added2 = _merge_new_rows(merged_df, df2)
        print(f"  Added {added2}/{len(df2)} rows from {run2.name} ({run2.id}).")

        merged_df.to_csv(csv_path, index=False)
        print(f"  Saved combined CSV to {csv_path} ({len(merged_df)} rows total, "
              f"{len(_score_keys_from_df(merged_df)[0])} classes).")

        plot_alignment_comparison(csv_path, inloop_csv_path, plot_path, title=dest_run.name)
        _upload_results(csv_path, plot_path, project, dest_run.id)
        print(f"  Uploaded combined results to {dest_run.name} ({dest_run.id}).")


_LATENT_DIM = 512
_IMAGE_H = _IMAGE_W = 224
_IMAGE_DIM = 3 * _IMAGE_H * _IMAGE_W  # 150 528


class LinearSynthesis(torch.nn.Module):
    """Synthetic linear generator G(w) = A @ w for unit-testing alignment scores.

    The Jacobian is exactly A everywhere (constant), so the tangent space is the
    k-dimensional column space of A.  For a uniformly random grad_x the expected
    alignment score is sqrt(k / d).
    """

    def __init__(self, d=_IMAGE_DIM, k=_LATENT_DIM, device='cpu'):
        super().__init__()
        self.register_buffer('A', torch.randn(d, k, device=device))
        self.k = k
        self.d = d

    def forward(self, w, noise_mode=None, force_fp32=None, **kwargs):
        # w arrives as [B, num_ws, k]; use only the first ws slot.
        w_flat = w[:, 0, :] if w.dim() == 3 else w   # [B, k]
        out = w_flat @ self.A.T                        # [B, d]
        return out.view(out.shape[0], 3, _IMAGE_H, _IMAGE_W)


def run_synthetic_test(num_samples=20, chunk_size=40, seed=None):
    """Verify compute_gradient_alignment_score against the known analytic expectation.

    For a fixed k-dimensional subspace in R^d and a uniformly random unit vector g,
    E[||P g|| / ||g||] = sqrt(k / d).  We use a linear generator whose Jacobian is
    constant (a random d×k matrix), so the tangent space dimension is exactly k=512
    and d = 3*224*224 = 150 528.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    k, d = _LATENT_DIM, _IMAGE_DIM
    expected = (k / d) ** 0.5

    print(f"\nSynthetic linear generator test")
    print(f"  k={k}, d={d}")
    print(f"  Expected alignment score: sqrt({k}/{d}) = {expected:.6f}")
    print(f"  num_samples={num_samples}, chunk_size={chunk_size}, device={device}")

    if seed is not None:
        torch.manual_seed(seed)
        print(f"  RNG seeded with {seed}.")

    synthesis = LinearSynthesis(d=d, k=k, device=device)
    num_ws = 1

    w_batch = torch.zeros(num_samples, 1, k, device=device)
    grad_x  = torch.randn(num_samples, 3, _IMAGE_H, _IMAGE_W, device=device)

    scores = compute_gradient_alignment_score(
        synthesis, w_batch, grad_x, num_ws,
        transformations=None, chunk_size=chunk_size,
    )

    score_vals = scores.tolist()
    mean_score = float(scores.mean())
    std_score  = float(scores.std())
    err = abs(mean_score - expected)

    print(f"\nResults:")
    for i, s in enumerate(score_vals):
        print(f"  sample {i:>2}: {s:.6f}")
    print(f"\n  Mean:     {mean_score:.6f}  (std={std_score:.6f})")
    print(f"  Expected: {expected:.6f}")
    print(f"  Error:    {err:.6f}  ({err / expected * 100:.2f}% of expected)")


def _load_alignment_csv(path):
    """Return a DataFrame with columns [batch, iteration, target_classes, sample, alignment_score]."""
    return pd.read_csv(path)


def plot_alignment_comparison(recalc_csv_path, inloop_csv_path, out_path, title=None):
    """Overlay histograms of post-hoc vs in-loop alignment scores and save to out_path.

    In-loop histogram is only added if inloop_csv_path exists.
    Returns out_path so the caller can upload it.
    """
    recalc_df = _load_alignment_csv(recalc_csv_path)
    recalc_scores = recalc_df['alignment_score'].to_numpy()

    has_inloop = os.path.exists(inloop_csv_path)
    inloop_df = _load_alignment_csv(inloop_csv_path) if has_inloop else None
    inloop_scores = inloop_df['alignment_score'].to_numpy() if inloop_df is not None else None

    bins = np.linspace(0, 1, 41)  # fixed [0,1] range; 40 bins of width 0.025

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(recalc_scores, bins=bins, alpha=0.6, label=f'post-hoc (n={len(recalc_scores)})',
            color='steelblue', edgecolor='white', linewidth=0.4)
    ax.axvline(recalc_scores.mean(), color='steelblue', linestyle='--', linewidth=1.5,
               label=f'post-hoc mean = {recalc_scores.mean():.3f}')

    if has_inloop and inloop_scores is not None and len(inloop_scores) > 0:
        ax.hist(inloop_scores, bins=bins, alpha=0.6, label=f'in-loop (n={len(inloop_scores)})',
                color='tomato', edgecolor='white', linewidth=0.4)
        ax.axvline(inloop_scores.mean(), color='tomato', linestyle='--', linewidth=1.5,
                   label=f'in-loop mean = {inloop_scores.mean():.3f}')

    ax.set_xlabel('Gradient-manifold alignment score')
    ax.set_ylabel('Count')
    plot_title = f'Alignment scores: {title}' if title else 'Alignment score distribution: post-hoc vs in-loop'
    ax.set_title(plot_title)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot: {out_path}")
    if not has_inloop:
        print("  (no in-loop CSV found — only post-hoc scores plotted)")
    return out_path


def compute_grad_magnitudes_for_run(run_id, project, classes=None):
    """Plot a histogram of inversion-time grad_x L2 norms and log summary metrics to W&B.

    Logs:
      - avg_grad_magnitude          — mean over all entries / samples
      - avg_grad_magnitude_iter_{x} — mean at iteration x, only for iterations where every
                                      class present in the log has at least one entry
    Fails gracefully if the raw log file is not found for this run.
    """
    out_dir = "model_inversion/plug_and_play/results/alignment_score"
    os.makedirs(out_dir, exist_ok=True)
    classes_tag = f"_c{'_'.join(str(c) for c in sorted(classes))}" if classes is not None else ""
    plot_path = os.path.join(out_dir, f"{run_id}_grad_magnitude{classes_tag}.png")

    api = wandb.Api()
    run = api.run(f"{ENTITY}/{project}/{run_id}")
    run_name = run.name

    raw_log_path = f"model_inversion/plug_and_play/results/alignment_raw/{run_id}.pt"
    try:
        downloaded = run.file(raw_log_path).download(exist_ok=True)
    except Exception as e:
        print(f"  [{run_id}] No raw log found — skipping. ({e})")
        return

    raw_log = torch.load(downloaded.name, map_location='cpu', weights_only=False)
    print(f"Raw log: {len(raw_log)} entries, iterations {sorted(set(e['iteration'] for e in raw_log))}")

    if classes is not None:
        classes_set = set(classes)
        raw_log = [e for e in raw_log if any(c in classes_set for c in e['target_classes'])]
        print(f"Filtered to {len(raw_log)} entries for classes {sorted(classes_set)}.")

    if not raw_log:
        print(f"  [{run_id}] No entries after filtering — skipping.")
        return

    # Compute per-sample L2 norms and group by iteration
    all_magnitudes = []
    iter_magnitudes = {}   # iteration -> list of per-sample norms
    iter_classes = {}      # iteration -> set of classes seen (for completeness check)
    all_classes = set()

    for entry in raw_log:
        grad_x = entry['grad_x']                                    # [B, 3, H, W]
        norms = grad_x.view(grad_x.shape[0], -1).norm(dim=1).tolist()
        all_magnitudes.extend(norms)
        it = entry['iteration']
        iter_magnitudes.setdefault(it, []).extend(norms)
        iter_classes.setdefault(it, set()).update(entry['target_classes'])
        all_classes.update(entry['target_classes'])

    # Only report per-iteration averages for iterations where every class is covered
    complete_iters = sorted(it for it, cls in iter_classes.items() if cls >= all_classes)

    avg_grad_magnitude = float(np.mean(all_magnitudes))
    iter_avgs = {it: float(np.mean(iter_magnitudes[it])) for it in complete_iters}

    print(f"avg_grad_magnitude: {avg_grad_magnitude:.6f}")
    for it, avg in iter_avgs.items():
        print(f"  avg_grad_magnitude_iter_{it}: {avg:.6f}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(all_magnitudes, bins=50, color='steelblue', edgecolor='white', linewidth=0.4, alpha=0.8)
    ax.axvline(avg_grad_magnitude, color='steelblue', linestyle='--', linewidth=1.5,
               label=f'mean = {avg_grad_magnitude:.4f}')
    ax.set_xlabel('Gradient magnitude (L2 norm)')
    ax.set_ylabel('Count')
    ax.set_title(f'Inversion gradient magnitudes: {run_name}')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved gradient magnitude plot: {plot_path}")

    # Upload
    wandb_run = wandb.init(project=project, entity=ENTITY, id=run_id, resume="must")
    wandb.summary['avg_grad_magnitude'] = avg_grad_magnitude
    for it, avg in iter_avgs.items():
        wandb.summary[f'avg_grad_magnitude_iter_{it}'] = avg
    wandb.save(plot_path)
    print(f"Logged metrics and uploaded plot to W&B run {run_id} ({wandb_run.name}).")
    wandb_run.finish()


def _log_alignment_metrics(recalc_df, inloop_df, plot_path):
    """Log alignment summary scalars and histogram image to the active W&B run."""
    wandb.summary['avg_alignment_score_posthoc'] = float(recalc_df['alignment_score'].mean())
    for it, grp in recalc_df.groupby('iteration'):
        wandb.summary[f'avg_alignment_score_posthoc_iter_{it}'] = float(grp['alignment_score'].mean())

    if inloop_df is not None and len(inloop_df) > 0:
        wandb.summary['avg_alignment_score_inloop'] = float(inloop_df['alignment_score'].mean())
        for it, grp in inloop_df.groupby('iteration'):
            wandb.summary[f'avg_alignment_score_inloop_iter_{it}'] = float(grp['alignment_score'].mean())

    if os.path.exists(plot_path):
        wandb.summary['alignment_score_histogram'] = wandb.Image(plot_path)


ALIGN_OUT_DIR = "model_inversion/plug_and_play/results/alignment_score"


def _recalc_paths(run_id, det_transforms, seed):
    """Paths of the single accumulating post-hoc CSV / plot for a run (not keyed by class set)."""
    transforms_tag = "det" if det_transforms else "stoch"
    suffix = f"_recalc_{transforms_tag}_s{seed}" if seed is not None else f"_recalc_{transforms_tag}"
    return (os.path.join(ALIGN_OUT_DIR, f"{run_id}{suffix}.csv"),
            os.path.join(ALIGN_OUT_DIR, f"{run_id}.csv"),
            os.path.join(ALIGN_OUT_DIR, f"{run_id}{suffix}_comparison.png"))


def _score_keys_from_df(df):
    """Return (classes, (batch, iteration, frozenset(classes)) keys) present in a recalc CSV DataFrame."""
    classes, keys = set(), set()
    for batch, iteration, tc in zip(df['batch'], df['iteration'], df['target_classes']):
        tcs = frozenset(int(c) for c in ast.literal_eval(tc))
        classes.update(tcs)
        keys.add((batch, iteration, tcs))
    return classes, keys


def _existing_score_keys(csv_path):
    return _score_keys_from_df(pd.read_csv(csv_path))


def _merge_new_rows(base_df, new_df):
    """Concatenate new_df onto base_df, dropping any new_df row whose (batch, iteration,
    target_classes) key is already present in base_df. Returns (merged_df, num_rows_added)."""
    if base_df is None:
        return new_df.reset_index(drop=True), len(new_df)
    _, base_keys = _score_keys_from_df(base_df)
    keep = [(batch, iteration, frozenset(ast.literal_eval(tc))) not in base_keys
            for batch, iteration, tc in zip(new_df['batch'], new_df['iteration'], new_df['target_classes'])]
    merged = pd.concat([base_df, new_df[keep]], ignore_index=True)
    return merged, sum(keep)


_CSV_CHOICES = {}  # run_id -> chosen W&B path, so an ambiguous run is only prompted once


_LIST_FILES_RETRIES = 3
_LIST_FILES_RETRY_DELAY = 2.0  # seconds

_FILES_SUPPORTS_PATTERN = None  # cached across calls; set on first use


def _files_supports_pattern(run):
    """Whether this wandb SDK's Run.files() accepts a `pattern` kwarg (added in a later
    release than the 0.18.6 this repo pins). Checked via reflection, not a version string,
    so it stays correct regardless of what's actually installed."""
    global _FILES_SUPPORTS_PATTERN
    if _FILES_SUPPORTS_PATTERN is None:
        try:
            _FILES_SUPPORTS_PATTERN = 'pattern' in inspect.signature(run.files).parameters
        except (TypeError, ValueError):
            _FILES_SUPPORTS_PATTERN = False
    return _FILES_SUPPORTS_PATTERN


def _list_recalc_csvs(run):
    """W&B paths of the run's post-hoc (recalc) alignment CSVs, excluding the in-loop {run_id}.csv.

    When the installed wandb SDK supports it, queries with a server-side `pattern` scoped to
    ALIGN_OUT_DIR instead of paginating the run's full file list (which can be large -- media,
    artifacts, etc.). Besides being faster, this sidesteps a W&B pagination bug where a run with
    exactly a multiple of the page size (50) worth of files can make every page request past the
    boundary come back with a null 'run' node, crashing the SDK's paginator on every attempt (not
    just transiently). Falls back to listing all files (with retries) on older SDKs.
    """
    prefix = ALIGN_OUT_DIR + "/"
    kwargs = {"pattern": f"{prefix}%.csv"} if _files_supports_pattern(run) else {}
    names = None
    for attempt in range(1, _LIST_FILES_RETRIES + 1):
        try:
            names = [f.name for f in run.files(**kwargs)]
            break
        except TypeError:
            # Genuine transient glitch: the GraphQL response's file list came back null mid-page.
            # Retry a few times before giving up, so a flaky response doesn't get mistaken for
            # "no alignment CSV".
            if attempt < _LIST_FILES_RETRIES:
                print(f"  [{run.id}] transient error listing files on W&B "
                      f"(attempt {attempt}/{_LIST_FILES_RETRIES}) — retrying...")
                time.sleep(_LIST_FILES_RETRY_DELAY)
            else:
                print(f"  [{run.id}] could not list files on W&B after {_LIST_FILES_RETRIES} attempts "
                      "— treating as no alignment CSV.")
                return []
    return sorted(n for n in names
                  if n.startswith(prefix) and n.endswith(".csv")
                  and os.path.basename(n) != f"{run.id}.csv"
                  and "_recalc_" in os.path.basename(n))


def _fetch_recalc_csv(run):
    """Download the run's post-hoc alignment CSV from W&B (overwriting any local copy).

    Accepts any recalc CSV in the alignment_score folder regardless of suffix (transforms tag,
    seed, class list). If there are several, the user is prompted to choose one (remembered per
    run). Returns the local path, or None if the run has none.
    """
    if run.id not in _CSV_CHOICES:
        paths = _list_recalc_csvs(run)
        if not paths:
            return None
        if len(paths) == 1:
            _CSV_CHOICES[run.id] = paths[0]
        else:
            options = []
            for path in paths:
                run.file(path).download(replace=True)
                df = pd.read_csv(path)
                options.append(f"{os.path.basename(path)}  ({len(_score_keys_from_df(df)[0])} classes, {len(df)} rows)")
            _, idx = pick(options, f"Multiple alignment CSVs for {run.name} ({run.id}); choose one:")
            _CSV_CHOICES[run.id] = paths[idx]
    path = _CSV_CHOICES[run.id]
    run.file(path).download(replace=True)
    return path if os.path.exists(path) else None


def _fetch_num_csv_classes(run):
    """Number of classes covered by the run's W&B alignment CSV, or None if it has none."""
    csv_path = _fetch_recalc_csv(run)
    if csv_path is None:
        return None
    classes, _ = _existing_score_keys(csv_path)
    return len(classes)


def _start_align_run(run_id, run_name, source_project, args_config):
    """Start a separate W&B run in ALIGN_PROJECT for this run's post-hoc alignment calculation."""
    # print("Sleeping for 10min...")
    # time.sleep(600)
    return wandb.init(
        project=ALIGN_PROJECT, entity=ENTITY, name=f"{run_name}_align",
        config={"source_run_id": run_id, "source_run_name": run_name,
                "source_project": source_project, **args_config},
        reinit=True,
    )


def _upload_results(csv_path, plot_path, project, run_id, align_run=None):
    """Log results to the align_scores run (if given), then to the original attack run."""
    out_dir = os.path.dirname(csv_path)
    inloop_csv_path = os.path.join(out_dir, f"{run_id}.csv")
    recalc_df = _load_alignment_csv(csv_path)
    inloop_df = _load_alignment_csv(inloop_csv_path) if os.path.exists(inloop_csv_path) else None

    if align_run is not None:
        wandb.save(csv_path)
        wandb.save(plot_path)
        _log_alignment_metrics(recalc_df, inloop_df, plot_path)
        print(f"Logged results to align run {align_run.name} ({align_run.id}) in {ALIGN_PROJECT}.")
        align_run.finish()

    wandb_run = wandb.init(project=project, entity=ENTITY, id=run_id, resume="must")
    wandb.save(csv_path)
    wandb.save(plot_path)
    _log_alignment_metrics(recalc_df, inloop_df, plot_path)
    print(f"Uploaded {csv_path} and {plot_path} to W&B run {run_id} ({wandb_run.name}).")
    wandb_run.finish()


def _log_score_to_align_run(row, done, total):
    batch, iteration, target_classes, sample, score = row
    wandb.log({"alignment_score": score, "batch": batch, "iteration": iteration,
               "sample": sample, "scores_done": done, "scores_total": total}, step=done)


def _pick_prior_run(run, project, det_transforms, seed):
    """Prompt for an optional other run whose existing W&B alignment CSV is combined into `run`'s.

    Candidates are other PPA-main runs on the same target model (created since MIN_RUN_DATE) that
    have an alignment CSV on W&B. Returns (prior_run, prior_df) or (None, None) if skipped.
    """
    api = wandb.Api()
    target_id = run.config.get("target_wandb_id")
    filters = {"config.target_wandb_id": target_id} if target_id is not None else {}
    candidates = []
    for r in api.runs(f"{ENTITY}/{project}", filters=filters):
        if r.id == run.id or str(r.created_at) < MIN_RUN_DATE:
            continue
        n = _fetch_num_csv_classes(r)
        if n is not None:
            candidates.append((r, n))
    if not candidates:
        print(f"  No other runs with an alignment CSV to combine into {run.name} ({run.id}).")
        return None, None
    candidates.sort(key=lambda x: x[0].created_at)
    options = [f"[{n} classes]  {r.name}  ({r.id})  [{r.state}]" for r, n in candidates]
    options.append("(none — don't combine with another run's CSV)")
    _, idx = pick(options, f"Choose an additional run whose alignment CSV is combined into "
                           f"{run.name} ({run.id}):")
    if idx >= len(candidates):
        return None, None
    prior_run = candidates[idx][0]
    return prior_run, pd.read_csv(_fetch_recalc_csv(prior_run))


def compute_alignment_scores_for_run(run_id, chunk_size=None, max_samples=None, project=None, seed=None, det_transforms=True, use_run_config=False, classes=None, prior_df=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    os.makedirs(ALIGN_OUT_DIR, exist_ok=True)
    csv_path, inloop_csv_path, plot_path = _recalc_paths(run_id, det_transforms, seed)

    # Fetch run name for histogram title (lightweight call — just metadata)
    api = wandb.Api()
    run = api.run(f"{ENTITY}/{project}/{run_id}")
    run_name = run.name

    align_run = _start_align_run(run_id, run_name, project, {
        "chunk_size": chunk_size, "max_samples": max_samples, "seed": seed,
        "det_transforms": det_transforms, "use_run_config": use_run_config, "classes": classes,
    })

    # Download raw (w, grad_x) log artifact
    raw_log_path = f"model_inversion/plug_and_play/results/alignment_raw/{run_id}.pt"
    downloaded = run.file(raw_log_path).download(exist_ok=True)
    print(f"Downloaded raw log: {downloaded.name}")
    raw_log = torch.load(downloaded.name, map_location='cpu', weights_only=False)
    print(f"Raw log: {len(raw_log)} entries "
          f"({len(set(e['batch_idx'] for e in raw_log))} batches, "
          f"iterations {sorted(set(e['iteration'] for e in raw_log))})")

    if classes is not None:
        classes_set = set(classes)
        raw_log = [e for e in raw_log if any(c in classes_set for c in e['target_classes'])]
        print(f"Filtered to {len(raw_log)} entries for classes {sorted(classes_set)}.")

    # Skip work already present in an existing CSV: only score classes not yet covered
    fetched_csv = _fetch_recalc_csv(run)
    existing_df = None
    if fetched_csv is not None:
        # may be the stoch CSV when det was requested; keep extending/uploading that same file
        csv_path = fetched_csv
        plot_path = csv_path[:-len(".csv")] + "_comparison.png"
        existing_df = pd.read_csv(csv_path)

    # Snapshot this path's state on W&B now, so a concurrent process writing to the same
    # path while we're computing (a long JVP+SVD pass) can be detected before we save.
    start_csv_updated_at = run.file(csv_path).updatedAt

    # Combine with scores from another run: add only rows whose classes aren't already present
    if prior_df is not None:
        have = _score_keys_from_df(existing_df)[0] if existing_df is not None else set()
        keep = [not (have & {int(c) for c in ast.literal_eval(tc)}) for tc in prior_df['target_classes']]
        seed_rows = prior_df[keep]
        if len(seed_rows) > 0:
            existing_df = pd.concat([existing_df, seed_rows], ignore_index=True) if existing_df is not None else seed_rows.copy()
            existing_df.to_csv(csv_path, index=False)
            print(f"Added {len(seed_rows)} rows from the other run into {csv_path}.")

    if existing_df is not None:
        existing_classes, existing_keys = _score_keys_from_df(existing_df)
        requested = set(classes) if classes is not None else {c for e in raw_log for c in e['target_classes']}
        new_classes = requested - existing_classes
        print(f"Existing CSV covers {len(existing_classes)} classes; "
              f"{len(new_classes)} requested classes are new: {sorted(new_classes)}")
        raw_log = [e for e in raw_log
                   if any(c in new_classes for c in e['target_classes'])
                   and (e['batch_idx'], e['iteration'], frozenset(e['target_classes'])) not in existing_keys]
        print(f"{len(raw_log)} raw log entries left to score.")
        if not raw_log:
            print("Nothing new to compute — re-logging existing results.")
            plot_alignment_comparison(csv_path, inloop_csv_path, plot_path, title=run_name)
            _upload_results(csv_path, plot_path, project, run_id, align_run=align_run)
            return

    # Load attack config from W&B
    attack_config, run_name = get_config(ENTITY, project, run_id)
    if "model" in attack_config:
        attack_config.attack = attack_config.model
    print(f"Loaded config for run '{run_name}' ({run_id})")

    # Resolve chunk_size / max_samples from the run's stored alignment_score config if requested
    run_as_cfg = OmegaConf.select(attack_config, 'attack.attack.alignment_score', default=None)
    if use_run_config and run_as_cfg is not None:
        run_chunk_size = OmegaConf.select(run_as_cfg, 'chunk_size', default=None)
        run_max_samples = OmegaConf.select(run_as_cfg, 'max_samples', default=None)
        chunk_size = run_chunk_size if run_chunk_size is not None else chunk_size
        max_samples = run_max_samples
        print(f"Using run config values: chunk_size={chunk_size}, max_samples={max_samples}")
    elif use_run_config:
        print("No alignment_score config found in W&B run — falling back to provided values.")
    print(f"chunk_size={chunk_size}, max_samples={max_samples}")

    # Load target config so convert_configs can build a full AttackConfigParser
    target_config, _ = get_target_config_and_weights(attack_config)

    new_attack_config_path = ppa_modify.convert_configs(target_config, attack_config)
    new_attack_config = AttackConfigParser(new_attack_config_path)
    transformations = new_attack_config.create_attack_transformations()
    if det_transforms:
        scored_transformations = make_det_transformations(transformations)
        if transformations is not None:
            _all = transformations.transforms if hasattr(transformations, 'transforms') else [transformations]
            _kept_ids = set(id(t) for t in (scored_transformations.transforms if hasattr(scored_transformations, 'transforms') else ([scored_transformations] if scored_transformations else [])))
            _removed = [type(t).__name__ for t in _all if id(t) not in _kept_ids]
            if _removed:
                print(f"Using det_transformations: removed stochastic {_removed}")
    else:
        scored_transformations = transformations
        print("Using full transformations (stochastic transforms retained).")

    # Load generator synthesis network
    stylegan_model = attack_config['attack']['stylegan_model']
    G = load_generator(stylegan_model, device)
    num_ws = G.num_ws
    synthesis = torch.nn.DataParallel(G.synthesis, device_ids=gpu_devices)
    synthesis.num_ws = num_ws
    del G

    # Compute alignment scores
    if seed is not None:
        torch.manual_seed(seed)
        print(f"RNG seeded with {seed}.")
    print(f"Computing alignment scores (chunk_size={chunk_size}, max_samples={max_samples})...")
    rows = compute_alignment_scores_from_raw_log(
        raw_log, synthesis, num_ws,
        transformations=scored_transformations,
        chunk_size=chunk_size,
        max_samples=max_samples,
        device=device,
        on_score=_log_score_to_align_run,
    )
    print(f"Done. {len(rows)} score entries computed.")

    # Guard against a concurrent process having written to this same CSV path on W&B while we
    # were computing -- re-check its state now and, if it changed, save under "_v1" (or the next
    # free "_vN") instead of silently overwriting whatever it wrote.
    current_csv_updated_at = run.file(csv_path).updatedAt
    if current_csv_updated_at != start_csv_updated_at:
        print(f"WARNING: {csv_path} on W&B changed since this run started "
              f"({start_csv_updated_at!r} -> {current_csv_updated_at!r}) -- a concurrent process "
              "must have written to it. Saving to a separate versioned file instead.")
        base, ext = os.path.splitext(csv_path)
        version = 1
        versioned_csv_path = f"{base}_v{version}{ext}"
        while os.path.exists(versioned_csv_path):
            version += 1
            versioned_csv_path = f"{base}_v{version}{ext}"
        csv_path = versioned_csv_path
        plot_path = csv_path[:-len(".csv")] + "_comparison.png"

    # Save CSV locally (appending to any existing scores)
    columns = ['batch', 'iteration', 'target_classes', 'sample', 'alignment_score']
    new_df = pd.DataFrame(rows, columns=columns)
    full_df = pd.concat([existing_df, new_df], ignore_index=True) if existing_df is not None else new_df
    full_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} ({len(new_df)} new rows, {len(full_df)} total)")

    # Plot comparison histogram (post-hoc vs in-loop if available)
    plot_alignment_comparison(csv_path, inloop_csv_path, plot_path, title=run_name)
    _upload_results(csv_path, plot_path, project, run_id, align_run=align_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-hoc PPA alignment score computation from saved raw log."
    )
    parser.add_argument('--chunk_size', type=int, default=40,
                        help="JVP chunk size (latent directions per forward pass). Default: 40.")
    parser.add_argument('--max_samples', type=int, default=2,
                        help="Max samples per batch entry to score (default: 2).")
    parser.add_argument('--project', type=str, default=DEFAULT_PROJECT,
                        help=f"W&B project containing the source run. Default: {DEFAULT_PROJECT}")
    parser.add_argument('--seed', type=int, default=None,
                        help="Seed for PyTorch RNG before scoring. "
                             "Omit to use the current RNG state (non-deterministic across runs).")
    parser.add_argument('--no-det-transforms', dest='det_transforms', action='store_false',
                        help="Use full attack transformations (including RandomResizedCrop) instead of "
                             "the deterministic subset (CenterCrop + Resize only). Useful for comparing "
                             "against old behaviour or investigating the effect of stochastic transforms.")
    parser.add_argument('--num-samples', type=int, default=20,
                        help="Number of random gradient vectors for the synthetic test. Default: 20.")
    parser.add_argument('--classes', type=lambda s: [int(c) for c in s.split(',')], default=None,
                        help="Restrict scoring to entries whose target_classes intersect this set "
                             "(e.g. --classes 3,7,12). Default: all classes.")
    parser.add_argument('--required-classes', type=int, default=15,
                        help="For the 'list runs missing full-class CSVs' mode: the number of "
                             "classes a run's alignment CSV must cover to not be listed. Default: 15.")
    parser.add_argument('--align15-csv-out', type=str, default=DEFAULT_ALIGN15_CSV_OUT,
                        help="For the 'list runs missing full-class CSVs' mode: where to save the "
                             f"target_wandb_id -> 15-class mean alignment score CSV. Default: {DEFAULT_ALIGN15_CSV_OUT}")
    parser.set_defaults(det_transforms=True)
    args = parser.parse_args()

    mode, _ = pick(
        ["Run post-hoc evaluation", "Plot gradient magnitude histograms",
         "List runs missing full-class alignment CSVs",
         "Combine two runs' alignment CSVs into a third run",
         "Run synthetic linear generator test"],
        "Select mode:",
    )

    if mode == "List runs missing full-class alignment CSVs":
        list_runs_missing_full_classes(project=args.project, required_classes=args.required_classes,
                                       align15_csv_out=args.align15_csv_out)

    elif mode == "Combine two runs' alignment CSVs into a third run":
        combine_runs_into_third(project=args.project, det_transforms=args.det_transforms, seed=args.seed)

    elif mode == "Run post-hoc evaluation":
        prior_for = {}
        runs = pick_runs_to_score(project=args.project, csv_status=True,
                                  det_transforms=args.det_transforms, seed=args.seed,
                                  priors_out=prior_for)
        if not runs:
            print("No runs selected, exiting.")
            sys.exit(0)
        run_ids = [r.id for r in runs]

        config_source, _ = pick(
            [f"Use each run's stored alignment_score config",
             f"Use CLI values  (chunk_size={args.chunk_size}, max_samples={args.max_samples})"],
            "Which alignment_score config to use?",
        )
        use_run_config = config_source.startswith("Use each run's")

        print("\nExisting post-hoc alignment CSVs (from W&B):")
        for r in runs:
            existing_csv = _fetch_recalc_csv(r)
            if existing_csv is not None:
                covered, _ = _existing_score_keys(existing_csv)
                print(f"  {r.id}: found, covers classes {sorted(covered)}")
            else:
                print(f"  {r.id}: none (will score from scratch)")

        for run_id in run_ids:
            print(f"\n{'='*60}\n  {run_id}\n{'='*60}")
            prior_run, prior_df = prior_for[run_id]
            if prior_run is not None:
                print(f"Combining with CSV from {prior_run.name} ({prior_run.id}): {len(prior_df)} rows.")
            compute_alignment_scores_for_run(
                run_id=run_id,
                chunk_size=args.chunk_size,
                max_samples=args.max_samples,
                project=args.project,
                seed=args.seed,
                det_transforms=args.det_transforms,
                use_run_config=use_run_config,
                classes=args.classes,
                prior_df=prior_df,
            )

    elif mode == "Plot gradient magnitude histograms":
        runs = pick_runs_with_preset(project=args.project)
        if not runs:
            print("No runs selected, exiting.")
            sys.exit(0)
        for run in runs:
            print(f"\n{'='*60}\n  {run.id}\n{'='*60}")
            compute_grad_magnitudes_for_run(
                run_id=run.id,
                project=args.project,
                classes=args.classes,
            )

    else:
        run_synthetic_test(
            num_samples=args.num_samples,
            chunk_size=args.chunk_size,
            seed=args.seed,
        )