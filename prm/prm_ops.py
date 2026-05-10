"""
PRM core operations: proximity regularization, write-back, and consolidation.

This module implements the core mechanics of the PRM algorithm:
  - Proximity loss: ||w_t - w_{t-1}||^2 regularization on LoRA params
  - Parameter snapshot: capture LoRA state before training for the proximity anchor
  - Write-back: merge fast branch into shared model (fixed, theory, budgeted)
  - Consolidation: post-merge conservative re-training (Stage 3)
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler


def get_lora_params_snapshot(model):
    """Snapshot current LoRA parameters for use as proximity anchor.

    Called at the start of Stage 1 (fast proposal) to capture the
    current LoRA state w_{t-1}. The proximity loss then penalizes
    deviation from this snapshot during training.

    Args:
        model: The ViT model with LoRA adapters.

    Returns:
        dict: {param_name: param_tensor_clone} for all trainable LoRA params.
    """
    snapshot = {}
    for name, p in model.named_parameters():
        if "lora_" in name and p.requires_grad:
            snapshot[name] = p.data.clone()
    return snapshot


def proximity_loss(model, snapshot):
    """Compute L2 proximity loss between current LoRA params and snapshot.

    Implements the regularization term:
        L_prox = sum_i ||w_i - w_i^{snapshot}||^2

    This encourages the fast branch to stay close to the previous solution,
    preventing catastrophic forgetting during the proposal stage.

    Args:
        model:    The ViT model with LoRA adapters.
        snapshot: Parameter snapshot from get_lora_params_snapshot().

    Returns:
        Scalar tensor: the total squared L2 distance.
    """
    # Initialize on the model's device so .item() / .backward() work even
    # when the snapshot keys do not match any current parameter (e.g. when
    # LoRA was re-installed for a new task before the snapshot was retaken).
    device = next(model.parameters()).device
    loss = torch.zeros((), device=device, dtype=torch.float32)
    for name, p in model.named_parameters():
        if name in snapshot:
            loss = loss + torch.sum((p - snapshot[name]) ** 2)
    return loss


def prm_write_back(model, alpha, task_id):
    """Stage 2: Write-back -- merge fast branch into shared model.

    For each LoRA module, records the merge coefficient alpha:
        theta_t = theta_{t-1} + alpha * (w_t - theta_{t-1})

    This is implemented via LoRALayer.cal_alpha(), which stores alpha_t
    and shifts the effective weight computation.

    For task 0, alpha is always 1.0 (first task fully written).
    For subsequent tasks, alpha is either fixed (PRM-fixed) or
    computed from Fisher information (PRM-theory).

    Args:
        model:   The ViT model with LoRA adapters.
        alpha:   Merge coefficient (scalar or tensor).
        task_id: Current task index.
    """
    from .lora import LoRALayer

    for module in model.modules():
        if isinstance(module, LoRALayer):
            if task_id == 0:
                # First task: store the initial LoRA weight and merge with alpha=1
                module.old_weight = (
                    module.lora_B[0].weight @ module.lora_A[0].weight
                ).detach()
                module.cal_alpha(1.0)
            else:
                module.cal_alpha(alpha)


# ---------------------------------------------------------------------------
# PRM-theory: Fisher-optimal per-task alpha
# ---------------------------------------------------------------------------

def compute_prm_theory_alpha(model, fisher, task_id, embed_dim=768):
    """Compute the Fisher-optimal merge alpha for PRM-theory.

    PRM-theory uses the same Fisher-based formula as P&M, but is applied
    within the PRM framework (after proximity-regularized fast proposal).

    The formula minimizes expected loss increase across all seen tasks:
        alpha* = -sum(F_i * delta * (W_{t-1} - W_accum - lora_i))
                 / sum(F_i * delta^2)

    where F_i is the Fisher diagonal for task i, delta is the current
    task's LoRA adapter, and W_{t-1} is the accumulated merged weight.

    Args:
        model:     The ViT model with LoRA adapters.
        fisher:    Fisher storage dict with accumulated diagonal.
        task_id:   Current task index.
        embed_dim: Model embedding dimension (default 768 for ViT-B).

    Returns:
        float: The Fisher-optimal merge alpha for this task.
    """
    from .lora import LoRALayer

    if task_id == 0:
        return 1.0

    ups = 0.0
    downs = 0.0

    for module in model.modules():
        if not isinstance(module, LoRALayer):
            continue

        name = f"{module.layer}"
        delta = (
            module.lora_B[task_id].weight @ module.lora_A[task_id].weight
        )

        # Reconstruct W_{t-1}: accumulated merged weight before this task
        W_t_1 = 0.0
        for i in range(task_id):
            W_t_1 += (
                module.alphas[i]
                * module.lora_B[i].weight
                @ module.lora_A[i].weight
            )

        weight = 0.0
        for i in range(task_id + 1):
            lora_i = module.lora_B[i].weight @ module.lora_A[i].weight

            if "k" in module.pos:
                f_k = fisher[name][i][embed_dim: embed_dim * 2, :].abs()
                ups += (W_t_1 - weight - lora_i) * f_k * delta
                downs += delta * f_k * delta

            if "v" in module.pos:
                f_v = fisher[name][i][embed_dim * 2: embed_dim * 3, :].abs()
                ups += (W_t_1 - weight - lora_i) * f_v * delta
                downs += delta * f_v * delta

            if i < task_id:
                weight += module.alphas[i] * lora_i

    alpha = float((-torch.sum(ups) / torch.sum(downs)).detach().cpu().item())
    return alpha


# ---------------------------------------------------------------------------
# PRM-budgeted: drift-constrained alpha
# ---------------------------------------------------------------------------

def compute_prm_budgeted_alpha(model, task_id, drift_budget=5.0, base_alpha=0.8):
    """Compute the drift-constrained alpha for PRM-budgeted.

    PRM-budgeted starts from the PRM-fixed base alpha and only reduces
    it further when the cumulative parameter drift would exceed a budget B.

        alpha_t = min(base_alpha, alpha_max)

    where alpha_max is the largest alpha such that:
        ||sum_{i<=t} alpha_i * delta_i||_F <= B

    The drift constraint prevents the shared model from wandering too
    far from the pretrained initialization, preserving transferability.

    Args:
        model:         The ViT model with LoRA adapters.
        task_id:       Current task index.
        drift_budget:  Maximum allowed Frobenius norm of total drift.
        base_alpha:    Upper bound for alpha (from PRM-fixed, default 0.8).

    Returns:
        float: The drift-constrained alpha for this task.
    """
    from .lora import LoRALayer

    if task_id == 0:
        return 1.0

    # Compute current accumulated drift: ||sum_{i<t} alpha_i * delta_i||
    current_drift_sq = 0.0
    # Compute cross term and new delta norm for the quadratic bound
    cross_term = 0.0
    delta_norm_sq = 0.0

    for module in model.modules():
        if not isinstance(module, LoRALayer):
            continue

        # Current accumulated merged weight (before this task)
        W_accum = torch.zeros_like(
            module.lora_B[0].weight @ module.lora_A[0].weight
        )
        for i in range(task_id):
            W_accum += (
                module.alphas[i]
                * module.lora_B[i].weight
                @ module.lora_A[i].weight
            )

        # New task's LoRA delta
        delta = (
            module.lora_B[task_id].weight @ module.lora_A[task_id].weight
        )

        current_drift_sq += torch.sum(W_accum ** 2).item()
        cross_term += torch.sum(W_accum * delta).item()
        delta_norm_sq += torch.sum(delta ** 2).item()

    # Solve quadratic: ||W_accum + alpha * delta||^2 <= budget^2
    # => delta_norm_sq * alpha^2 + 2*cross_term * alpha + (current_drift_sq - budget^2) <= 0
    B_sq = drift_budget ** 2
    a = delta_norm_sq
    b = 2.0 * cross_term
    c = current_drift_sq - B_sq

    if a < 1e-12:
        # Delta is essentially zero; any alpha works
        return 1.0

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        # Already over budget; use minimal alpha
        return 0.01

    sqrt_disc = discriminant ** 0.5
    alpha_max = (-b + sqrt_disc) / (2 * a)

    # Clamp: drift constraint can only REDUCE alpha below base_alpha,
    # never increase it.  alpha=1.0 would mean complete replacement
    # (catastrophic forgetting), so we cap at base_alpha.
    alpha_max = max(0.01, min(base_alpha, alpha_max))
    return alpha_max


# ---------------------------------------------------------------------------
# GatedPRM: writeability-gated selective merging
# ---------------------------------------------------------------------------

def compute_writeability_gate(
    model, fisher, task_id, late_snapshots=None,
    a=1.0, b=1.0, c=1.0, eps=1e-8, rescale_mean=True,
    embed_dim=768, verbose=True,
):
    """Compute per-parameter writeability gate for GatedPRM.

    The gate decides which components of the task vector delta_t = B_t @ A_t
    are worth writing into the shared model. It is applied element-wise at
    write-back time:
        W = sum_i alpha_i * (g_i * (B_i @ A_i))

    Three signals drive the gate:
      Utility   U_j = z_layer(log(|delta_j| + eps))            -- current task magnitude
      Conflict  C_j = z_layer(log(Ftilde_j + eps))             -- avg Fisher of prior tasks
      Stability S_j = z_layer(-log(Var_late_j + eps))          -- late-epoch temporal consistency

    Gate logit is  a*U - b*C + c*S, passed through sigmoid. If rescale_mean,
    the gate is divided by its layer-wise mean so the merge magnitude is
    controlled by alpha alone (gate answers "what", alpha answers "how much").

    Args:
        model:          ViT model with LoRA adapters.
        fisher:         Fisher storage dict (as from init_fisher_storage()).
        task_id:        Current task index.
        late_snapshots: Optional dict {layer_key: list[W_k]} from last K
                        epochs of Stage 1. Keys must be "<layer>_<pos>".
        a, b, c:        Coefficients for Utility / Conflict / Stability.
        eps:            Numerical epsilon for log and std.
        rescale_mean:   If True, rescale each layer's gate so mean(g)=1.
        embed_dim:      Embedding dim (K/V slice size).
        verbose:        If True, print distribution stats per LoRA layer.

    Returns:
        dict: {"<layer>_<pos>": gate_tensor_or_None}. For task_id=0 all
        values are None (equivalent to g=1).
    """
    from .lora import LoRALayer

    gates = {}
    if task_id == 0:
        for module in model.modules():
            if isinstance(module, LoRALayer):
                gates[f"{module.layer}_{module.pos}"] = None
        return gates

    stats_lines = []
    for module in model.modules():
        if not isinstance(module, LoRALayer):
            continue

        layer_name = f"{module.layer}"
        pos = module.pos
        key = f"{layer_name}_{pos}"

        delta = (
            module.lora_B[task_id].weight @ module.lora_A[task_id].weight
        ).detach()

        U_raw = torch.log(delta.abs() + eps)
        U_z = (U_raw - U_raw.mean()) / (U_raw.std() + eps)

        F_accum = torch.zeros_like(delta)
        for i in range(task_id):
            f_i = fisher[layer_name][i]
            if pos == "k":
                F_accum += f_i[embed_dim: 2 * embed_dim, :].abs()
            elif pos == "v":
                F_accum += f_i[2 * embed_dim: 3 * embed_dim, :].abs()
            elif pos == "q":
                F_accum += f_i[:embed_dim, :].abs()
        F_accum = F_accum / float(max(1, task_id))
        C_raw = torch.log(F_accum + eps)
        C_z = (C_raw - C_raw.mean()) / (C_raw.std() + eps)

        if (
            late_snapshots is not None
            and key in late_snapshots
            and len(late_snapshots[key]) > 1
        ):
            snaps = torch.stack(late_snapshots[key])  # (K, out, in)
            V = snaps.var(dim=0)
            S_raw = -torch.log(V + eps)
            S_z = (S_raw - S_raw.mean()) / (S_raw.std() + eps)
        else:
            S_z = torch.zeros_like(delta)

        logit = a * U_z - b * C_z + c * S_z
        gate = torch.sigmoid(logit)
        if rescale_mean:
            gate = gate / (gate.mean() + eps)

        gates[key] = gate.detach()

        if verbose:
            g_flat = gate.flatten()
            q = torch.quantile(
                g_flat, torch.tensor([0.25, 0.50, 0.75], device=g_flat.device)
            ).tolist()
            stats_lines.append(
                f"    [gate L{layer_name}/{pos}] "
                f"mean={gate.mean().item():.3f} std={gate.std().item():.3f} "
                f"min={gate.min().item():.3f} p25={q[0]:.3f} p50={q[1]:.3f} "
                f"p75={q[2]:.3f} max={gate.max().item():.3f} "
                f"frac>0.8={(gate > 0.8).float().mean().item():.3f} "
                f"frac<0.2={(gate < 0.2).float().mean().item():.3f}"
            )

    if verbose and stats_lines:
        print("  Gate distribution stats:")
        for line in stats_lines:
            print(line)

    return gates


def prm_write_back_gated(model, alpha, task_id, gates):
    """Stage 2 with per-parameter gates (GatedPRM variant of prm_write_back).

    For each LoRA layer, records alpha and the corresponding gate tensor.
    Task 0 writes with alpha=1 and no gate (identity), matching prm_write_back.

    Args:
        model:   ViT model with LoRA adapters.
        alpha:   Scalar merge coefficient computed from the base alpha method.
        task_id: Current task index.
        gates:   Dict from compute_writeability_gate.
    """
    from .lora import LoRALayer

    for module in model.modules():
        if isinstance(module, LoRALayer):
            key = f"{module.layer}_{module.pos}"
            if task_id == 0:
                module.old_weight = (
                    module.lora_B[0].weight @ module.lora_A[0].weight
                ).detach()
                module.cal_alpha_gated(1.0, None)
            else:
                module.cal_alpha_gated(alpha, gates.get(key, None))


# ---------------------------------------------------------------------------
# Direction-level GatedPRM: SVD of delta, gate per singular direction
# ---------------------------------------------------------------------------

def apply_direction_gate(
    model, fisher, task_id, late_snapshots=None,
    a=1.0, b=1.0, c=1.0, eps=1e-8, rescale_mean=True,
    embed_dim=768, verbose=True,
):
    """Direction-level writeability gate: SVD-decompose delta_t and scale per direction.

    For each LoRALayer, delta_t = B_t @ A_t is rank <= r (LoRA rank). Decomposing:
        delta_t = U_r @ diag(S_r) @ Vt_r

    we gate each direction k by g_k, then reconstruct:
        delta_gated = U_r @ diag(g * S) @ Vt_r

    and overwrite lora_B = U_r @ diag(g * S), lora_A = Vt_r. The subsequent
    write-back uses the gated delta with a scalar alpha (no per-element gate).

    Three signals (all at direction level, z-scored within each LoRA layer):
      Utility   U_k = log(S_k + eps)                            -- singular value
      Conflict  C_k = log((u_k^2)^T F_accum (v_k^2) + eps)      -- quadratic form in Fisher
      Stability S_k = -log(Var_l(u_k^T W_l v_k) + eps)          -- projected scalar variance

    task_id == 0 is a no-op (delta retained as-is).

    Args:
        model:          ViT model with LoRA adapters.
        fisher:         Fisher storage dict.
        task_id:        Current task index.
        late_snapshots: {layer_key: list[W]} snapshots from last K epochs.
        a, b, c:        Coefficients for U / C / S.
        eps:            Numerical epsilon.
        rescale_mean:   If True, rescale g so layer mean(g)=1.
        embed_dim:      QKV slice size.
        verbose:        Print per-layer direction-level stats.
    """
    from .lora import LoRALayer

    if task_id == 0:
        return

    stats_lines = []
    for module in model.modules():
        if not isinstance(module, LoRALayer):
            continue

        layer_name = f"{module.layer}"
        pos = module.pos
        key = f"{layer_name}_{pos}"

        with torch.no_grad():
            delta = (
                module.lora_B[task_id].weight @ module.lora_A[task_id].weight
            ).detach()

            # SVD (full_matrices=False -> economy SVD)
            U_full, S_full, Vt_full = torch.linalg.svd(delta, full_matrices=False)
            r = min(module.rank, S_full.shape[0])
            U_r = U_full[:, :r].contiguous()        # (out, r)
            S_r = S_full[:r].contiguous()           # (r,)
            Vt_r = Vt_full[:r, :].contiguous()      # (r, in)

            # Utility
            U_raw = torch.log(S_r + eps)
            U_z = (U_raw - U_raw.mean()) / (U_raw.std() + eps)

            # Conflict (Fisher quadratic form on direction k)
            F_accum = torch.zeros_like(delta)
            for i in range(task_id):
                f_i = fisher[layer_name][i]
                if pos == "k":
                    F_accum += f_i[embed_dim: 2 * embed_dim, :].abs()
                elif pos == "v":
                    F_accum += f_i[2 * embed_dim: 3 * embed_dim, :].abs()
                elif pos == "q":
                    F_accum += f_i[:embed_dim, :].abs()
            F_accum = F_accum / float(max(1, task_id))
            u_sq = U_r ** 2                         # (out, r)
            v_sq = (Vt_r.t()) ** 2                  # (in, r)
            C_mat = u_sq.t() @ F_accum @ v_sq       # (r, r); diag gives per-k
            C_raw = torch.diagonal(C_mat).clamp(min=0)
            C_raw = torch.log(C_raw + eps)
            C_z = (C_raw - C_raw.mean()) / (C_raw.std() + eps)

            # Stability: projected scalar variance across late snapshots
            if (
                late_snapshots is not None
                and key in late_snapshots
                and len(late_snapshots[key]) > 1
            ):
                # sigma_k^(l) = u_k^T W_l v_k (scalar per direction per snapshot)
                snaps = torch.stack(late_snapshots[key])                 # (K, out, in)
                # einsum: out_r=(K,r) -> for each l,k: sum_ij U_r[i,k] W_l[i,j] Vt_r[k,j]
                out_r = torch.einsum(
                    "ir,lij,rj->lr", U_r, snaps, Vt_r,
                )                                                         # (K, r)
                var_k = out_r.var(dim=0)                                  # (r,)
                S_raw = -torch.log(var_k + eps)
                S_z = (S_raw - S_raw.mean()) / (S_raw.std() + eps)
            else:
                S_z = torch.zeros_like(S_r)

            logit = a * U_z - b * C_z + c * S_z
            g = torch.sigmoid(logit)
            if rescale_mean:
                g = g / (g.mean() + eps)

            # Overwrite lora_A, lora_B so that B@A = gated delta.
            # new_A = Vt_r (r, in); new_B = U_r @ diag(g * S_r) (out, r)
            new_A = Vt_r
            gS = g * S_r                                                  # (r,)
            new_B = U_r * gS.unsqueeze(0)                                 # (out, r)
            module.lora_A[task_id].weight.data.copy_(new_A)
            module.lora_B[task_id].weight.data.copy_(new_B)

            if verbose:
                gvals = g.tolist()
                svals = S_r.tolist()
                stats_lines.append(
                    f"    [dirgate L{layer_name}/{pos}] "
                    f"g={['%.2f'%v for v in gvals]} "
                    f"sigma={['%.3f'%v for v in svals]}"
                )

    if verbose and stats_lines:
        print("  Direction gate (per-layer, rank-ordered):")
        for line in stats_lines[:4]:  # show first 4 layers; rest are similar
            print(line)
        if len(stats_lines) > 4:
            print(f"    ... ({len(stats_lines) - 4} more layers)")


# ---------------------------------------------------------------------------
# Stage 3: Consolidation (post-merge conservative re-training)
# ---------------------------------------------------------------------------

def consolidate(
    model, dataloader, criterion, args, task_id,
    consolidation_epochs=2, consolidation_lr=1e-4,
):
    """Stage 3: Conservative re-training after write-back.

    After merging the fast branch (Stage 2), the consolidated model may
    benefit from a few epochs of conservative fine-tuning on the current
    task. This uses a lower learning rate and does NOT use proximity
    regularization or perturbation (perturbation=0 always).

    The goal is to smooth out any discontinuities introduced by the
    discrete write-back step while keeping the model close to the
    merged state.

    Only the current task's LoRA adapter is trained (same as Stage 1),
    but with much smaller learning rate and no proximity loss.

    Args:
        model:                The ViT model after Stage 2 write-back.
        dataloader:           Training dataloader for the current task.
        criterion:            Loss function (CrossEntropyLoss).
        args:                 Training arguments (for use_amp, temperature).
        task_id:              Current task index.
        consolidation_epochs: Number of consolidation epochs (default 2).
        consolidation_lr:     Learning rate for consolidation (default 1e-4).
    """
    from .lora import LoRALayer

    # Re-enable training on current task's LoRA
    for name, p in model.named_parameters():
        p.requires_grad_(False)
        if "head" in name:
            p.requires_grad_(True)
        elif "lora_A" in name:
            if str(task_id) in name.split("lora_A.")[1]:
                p.requires_grad_(True)
        elif "lora_B" in name:
            if str(task_id) in name.split("lora_B.")[1]:
                p.requires_grad_(True)

    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.training = True

    model.train()
    # Only set LoRA and head modules to train mode
    model.eval()
    for n, m in model.named_modules():
        if any(s in n for s in ("lora", "head")) and not isinstance(m, nn.Identity):
            m.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=consolidation_lr, weight_decay=1e-5,
    )
    amp_scaler = GradScaler(enabled=args.use_amp)

    for epoch in range(1, consolidation_epochs + 1):
        total_loss = 0.0
        total_samples = 0
        for images, target in dataloader:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=args.use_amp
            ):
                # No perturbation during consolidation
                logits = model(images, perturbation=0)

            loss = criterion(logits / args.temperature, target)

            optimizer.zero_grad()
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        avg_loss = total_loss / total_samples
        print(f"    Consolidation epoch [{epoch}/{consolidation_epochs}] loss={avg_loss:.4f}")

    # Return LoRA to inference mode
    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.training = False
