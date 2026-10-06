"""Gradient-manifold alignment score (Peng et al., "Generative Model Inversion Through
the Lens of the Manifold Hypothesis", NeurIPS'25; https://arxiv.org/abs/2509.20177).

For a synthesized image x = G(w), the score measures what fraction of the inversion
loss gradient dL/dx lies in the tangent space of the generator manifold at x:

    AS = cos(phi) = || P_x * grad_x || / || grad_x ||

where P_x = U_k U_k^T is the orthogonal projector onto Range(J_G(w)), and U_k is
obtained via SVD of the generator Jacobian J_G(w) (built column-by-column with JVPs,
since the full Jacobian is too large to materialize directly at image resolution).

Ported from the official AlignMI implementation's `compute_proj_x_orthogonalized`
(high_resolution/attacks/optimize.py), generalized to not depend on an Optimization
instance so it can be called from any attack's optimize loop.
"""

import torch
from torch.autograd.functional import jvp
from torchvision import transforms as T
from tqdm import tqdm

_STOCHASTIC_TRANSFORMS = (
    T.RandomResizedCrop, T.RandomCrop, T.RandomHorizontalFlip, T.RandomVerticalFlip,
    T.ColorJitter, T.RandomRotation, T.RandomAffine, T.RandomGrayscale, T.GaussianBlur,
    T.RandomPerspective, T.AutoAugment, T.RandAugment, T.TrivialAugmentWide,
)


def make_det_transformations(transformations):
    """Return a version of `transformations` with all stochastic augmentations removed.

    Used to ensure alignment score computation (both grad_x and Jacobian) is deterministic.
    Returns None if all transforms were stochastic or transformations is None.
    """
    if transformations is None:
        return None
    ts = transformations.transforms if isinstance(transformations, T.Compose) else [transformations]
    det = [t for t in ts if not isinstance(t, _STOCHASTIC_TRANSFORMS)]
    if not det:
        return None
    return T.Compose(det) if len(det) > 1 else det[0]


def compute_gradient_alignment_score(synthesis, w_batch, grad_x, num_ws, transformations=None, chunk_size=40, max_samples=None,
                                      mid_vector_batch=None, layer=None, end_layer=None):
    """Per-sample gradient-manifold alignment score.

    Args:
        synthesis: the generator's synthesis network (e.g. StyleGAN2 `G.synthesis`).
        w_batch: latent codes, shape [B, 1, latent_dim] or [B, num_ws, latent_dim].
        grad_x: dL/dx, shape [B, C, H, W] (e.g. `imgs.grad` after `loss.backward()`
            with `imgs.retain_grad()` called beforehand).
        num_ws: number of style-layer copies the generator expects (`G.num_ws`).
        transformations: optional callable applied to synthesized images, matching
            whatever transform pipeline produced the gradient being analyzed.
        chunk_size: number of tangent directions per JVP batch (trades memory for speed).
        max_samples: if set and B > max_samples, randomly subsample this many images
            from the batch before computing scores. Useful for in-loop monitoring where
            batch_size is large and per-sample JVP+SVD cost is prohibitive.
        mid_vector_batch: IF-GMI only. Intermediate-layer activation batch (`layer_in` in
            IF-GMI's synthesis), shape [B, C, H, W], or None. When given, `mid_vector_batch[i]`
            is jointly optimized with `w_batch[i]` (this is how IF-GMI's `Optimization.intermediate`
            treats layers with start_layer > 0), so the manifold's tangent space is built from
            BOTH: the Jacobian spans latent_dim + mid_vector.numel() directions instead of just
            latent_dim. This makes per-sample cost scale with mid_vector's resolution, which can be
            far larger than latent_dim at deeper layers -- tune chunk_size/max_samples accordingly.
            None (the default) reproduces the original PPA-style w-only Jacobian.
        layer, end_layer: IF-GMI only. Required whenever either is given (independently of
            mid_vector_batch): IF-GMI's synthesis network is stateful (`synthesis.module.set_layer`
            selects which block range `forward` runs), so this must be set to the same
            (start_layer, config.intermediate['end']) used when the logged grad_x was produced,
            even for start_layer == 0 where mid_vector_batch is None. Ignored (and unnecessary)
            for a plain, stateless synthesis network such as PPA's.

    Returns:
        Tensor of shape [min(B, max_samples)] with per-sample alignment scores in [0, 1].
    """
    gen = synthesis.module if hasattr(synthesis, 'module') else synthesis

    if layer is not None or end_layer is not None:
        if layer is None or end_layer is None:
            raise ValueError("layer and end_layer must both be given (or both omitted).")
        gen.set_layer(layer, end_layer)

    B = grad_x.shape[0]
    if max_samples is not None and B > max_samples:
        indices = torch.randperm(B, device=w_batch.device)[:max_samples]
        w_batch = w_batch[indices]
        grad_x = grad_x[indices]
        if mid_vector_batch is not None:
            mid_vector_batch = mid_vector_batch[indices]
        B = max_samples

    latent_dim = w_batch.shape[-1]

    scores = []
    for i in range(B):
        w_i = w_batch[i].detach().requires_grad_()  # [rows, latent_dim]
        if w_i.shape[0] == 1 and num_ws > 1:
            w_i = w_i.unsqueeze(0).repeat(1, num_ws, 1)  # [1, num_ws, latent_dim]
        else:
            w_i = w_i.unsqueeze(0)  # [1, rows, latent_dim]

        if mid_vector_batch is not None:
            mid_i = mid_vector_batch[i].detach().unsqueeze(0)  # [1, C, H, W]
            eye = torch.eye(latent_dim + mid_i.numel(), device=w_batch.device)

            def flattened_g(_w, _mid):
                out = gen(_w, layer_in=_mid, noise_mode='const')
                if transformations:
                    out = transformations(out)
                return out.flatten(start_dim=1)
        else:
            mid_i = None
            eye = torch.eye(latent_dim, device=w_batch.device)

            def flattened_g(_w):
                out = gen(_w, noise_mode='const') #, force_fp32=True)
                if transformations:
                    out = transformations(out)
                return out.flatten(start_dim=1)

        tangent_chunks = []
        v_chunks = torch.split(eye, chunk_size, dim=0)
        for v_chunk in tqdm(v_chunks, desc=f'Alignment score JVP (sample {i})', leave=False):
            # v_chunk rows are standard basis vectors of the combined (w, mid_vector) space, so
            # each row's w-part and mid-part are never simultaneously nonzero -- this correctly
            # isolates the w-only and mid_vector-only Jacobian columns even for chunks that straddle
            # the latent_dim/mid_vector.numel() boundary.
            w_dirs = v_chunk[:, :latent_dim]
            w_chunk_expanded = w_dirs.unsqueeze(1).expand(-1, w_i.shape[1], -1)
            w_i_exp = w_i.expand(w_chunk_expanded.shape[0], *w_i.shape[1:])

            if mid_i is not None:
                mid_dirs = v_chunk[:, latent_dim:].reshape(-1, *mid_i.shape[1:])
                mid_i_exp = mid_i.expand(mid_dirs.shape[0], *mid_i.shape[1:])
                _, tangent_chunk = jvp(flattened_g, (w_i_exp, mid_i_exp), (w_chunk_expanded, mid_dirs))
            else:
                _, tangent_chunk = jvp(flattened_g, (w_i_exp,), (w_chunk_expanded,))
            tangent_chunks.append(tangent_chunk)

        J = torch.cat(tangent_chunks, dim=0).transpose(0, 1).contiguous()
        J = J.view(grad_x[i].numel(), -1)

        U, _, _ = torch.linalg.svd(J, full_matrices=False)

        real_grad = grad_x[i].reshape(-1)
        proj = U @ (U.T @ real_grad)

        norm_grad = torch.norm(real_grad)
        norm_proj = torch.norm(proj)
        score = (norm_proj / norm_grad) if norm_grad > 0 else torch.zeros((), device=real_grad.device)
        scores.append(score.detach())

    return torch.stack(scores)


def compute_alignment_scores_from_raw_log(raw_log, synthesis, num_ws, transformations=None, chunk_size=40, max_samples=None, device=None, end_layer=None, on_score=None):
    """Post-hoc alignment scoring from raw (w, grad_x) pairs saved during an attack.

    `raw_log` is the list saved unconditionally during optimization (see
    `Optimization.alignment_raw_log` in Plug_and_Play_Attacks/attacks/optimize.py and
    IF_GMI/attacks/optimize.py, and the corresponding
    `model_inversion/{plug_and_play,if_gmi}/results/alignment_raw/{run_id}.pt` artifact),
    independent of whether the in-loop score was also computed live. This lets the expensive
    JVP+SVD work be deferred to an offline pass instead of paying for it during the attack.

    Transparently handles both PPA-style entries (no 'mid_vector'/'layer' keys) and IF-GMI-style
    entries (which additionally carry 'layer', 'end_layer', and possibly a non-None 'mid_vector'
    for layer > 0) -- see `compute_gradient_alignment_score`'s mid_vector_batch/layer/end_layer
    docs for what these change about the Jacobian and why IF-GMI's stateful synthesis needs them.

    Args:
        raw_log: list of dicts with keys 'batch_idx', 'iteration', 'target_classes', 'w', 'grad_x',
            and, for IF-GMI entries, 'layer', 'end_layer', 'mid_vector' (may be None for layer 0)
            (e.g. loaded via `torch.load(".../alignment_raw/{run_id}.pt")`).
        synthesis, num_ws, transformations, chunk_size: see `compute_gradient_alignment_score`.
        device: device to run the computation on (defaults to synthesis's own device).
        end_layer: fallback used only for IF-GMI entries that predate the 'end_layer' key (i.e.
            don't carry it themselves) -- pass the run's `config.intermediate['end']`. Entries
            with their own 'end_layer' ignore this.
        on_score: optional callback `on_score(row, done, total)` invoked after each score is computed.

    Returns:
        list of [batch_idx, iteration, target_classes, sample_idx, alignment_score] rows —
        one row per sample (not per batch). sample_idx is 0-indexed within the scored subset
        (which may be a random subsample of the batch when max_samples is set).
    """
    if device is None:
        device = next(synthesis.parameters()).device

    def _num_scored(entry):
        b = entry['w'].shape[0]
        return min(b, max_samples) if max_samples is not None else b

    total = sum(_num_scored(e) for e in raw_log)
    print(f"{total} alignment scores to compute across {len(raw_log)} raw log entries.")

    rows = []
    for entry in raw_log:
        w = entry['w'].to(device)
        grad_x = entry['grad_x'].to(device)
        mid_vector = entry.get('mid_vector')
        if mid_vector is not None:
            mid_vector = mid_vector.to(device)
        layer = entry.get('layer')
        entry_end_layer = entry.get('end_layer', end_layer)
        if layer is not None and entry_end_layer is None:
            raise ValueError(
                f"Raw log entry (batch_idx={entry.get('batch_idx')}, layer={layer}) has no "
                "'end_layer' and none was supplied as a fallback; pass end_layer=config.intermediate['end']."
            )
        scores = compute_gradient_alignment_score(
            synthesis, w, grad_x, num_ws, transformations=transformations, chunk_size=chunk_size, max_samples=max_samples,
            mid_vector_batch=mid_vector, layer=layer, end_layer=entry_end_layer,
        )
        for sample_idx, score in enumerate(scores.tolist()):
            rows.append([entry['batch_idx'], entry['iteration'], entry['target_classes'], sample_idx, score])
            done = len(rows)
            print(f"[{done}/{total}, {total - done} left] batch={entry['batch_idx']} "
                  f"iteration={entry['iteration']} classes={entry['target_classes']} "
                  f"sample={sample_idx} alignment_score={score:.6f}", flush=True)
            if on_score is not None:
                on_score(rows[-1], done, total)

    return rows
