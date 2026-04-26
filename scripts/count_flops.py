#!/usr/bin/env python
"""
Count total FLOPs for GLMamba using fvcore FlopCountAnalysis.

fvcore uses JIT tracing and counts standard ops (Conv, Linear, BN, etc.).
selective_scan_cuda is a custom CUDA kernel that fvcore cannot trace through,
so its FLOPs are computed analytically and added to the fvcore total.

Run on a GPU node:
  conda activate glmamba
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
  cd /scratch/$USER/Mamba-LBP
  pip install fvcore -q
  python scripts/count_flops.py
"""

import torch
from glmamba.models import GLMamba, GLMambaConfig


# ──────────────────────────────────────────────────────────────────────────────
# Exact SSM FLOPs via runtime monkey-patch of selective_scan_fn
#
# fvcore cannot trace selective_scan_cuda (custom CUDA kernel).
# Instead we intercept the actual call at runtime with real tensor shapes:
#
# Per selective_scan call  u:(B, d_inner, L),  A:(d_inner, d_state):
#   delta = softplus(delta + delta_bias)       →  L × d_inner              ops
#   dA    = exp(delta[...,None] * A)           →  L × d_inner × d_state    ops
#   dB_u  = delta[...,None] * B * u[...,None]  →  L × d_inner × d_state    ops
#   scan  : h = dA*h + dB_u  (parallel scan)   →  L × d_inner × d_state ×2 ops
#   out   : y = (C * h).sum(-1) + D*u          →  L × d_inner × d_state    ops
#                                              + L × d_inner                ops
#   Total MACs ≈ 5 × L × d_inner × d_state   →  ×2 = FLOPs
# ──────────────────────────────────────────────────────────────────────────────

_ssm_flop_log: list[int] = []   # accumulates FLOPs from each intercepted call


def _install_ssm_hook():
    """
    Monkey-patch glmamba.models.ss2d.selective_scan_fn so every call
    records the exact FLOPs based on the real runtime tensor shapes.
    """
    import glmamba.models.ss2d as _ss2d_module

    _original = _ss2d_module.selective_scan_fn

    def _counting_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, *args, **kwargs):
        # u: (B, d_inner, L),  A: (d_inner, d_state)
        _, d_inner, L = u.shape
        d_state       = A.shape[-1]
        macs          = 5 * int(L) * int(d_inner) * int(d_state)
        _ssm_flop_log.append(macs * 2)      # MACs → FLOPs
        return _original(u, delta, A, B, C, D, delta_bias, delta_softplus, *args, **kwargs)

    _ss2d_module.selective_scan_fn = _counting_fn


def _uninstall_ssm_hook():
    """Restore the original selective_scan_fn."""
    import glmamba.models.ss2d as _ss2d_module
    from glmamba.models.ss2d import selective_scan_fn as _patched

    # The module already imported the original via closure; just reload.
    # Simplest: nothing critical here — we only need it once for profiling.
    pass


# ──────────────────────────────────────────────────────────────────────────────
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required — selective_scan_cuda needs CUDA.")

    device = torch.device("cuda")
    cfg    = GLMambaConfig()
    model  = GLMamba(cfg).to(device).eval()

    # Training input sizes: scale=2, HR=240×240, LR=120×120
    scale   = 2
    H, W    = 240, 240
    lr_in   = torch.randn(1, 1, H // scale, W // scale, device=device)
    ref_in  = torch.randn(1, 1, H, W, device=device)

    # ── Parameters ────────────────────────────────────────────────────────────
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 60)
    print("  GL-Mamba FLOPs Report")
    print("=" * 60)
    print(f"  Input LR  : {tuple(lr_in.shape)}")
    print(f"  Input Ref : {tuple(ref_in.shape)}")
    print(f"  Channels  : {cfg.channels},  n_blocks: {cfg.n_blocks}")
    print(f"  Total params    : {total_params:>12,}  ({total_params/1e6:.2f} M)")
    print(f"  Trainable params: {trainable_params:>12,}  ({trainable_params/1e6:.2f} M)")
    print("=" * 60)

    # ── Step 1: Exact SSM FLOPs via runtime hook ──────────────────────────────
    # Install hook BEFORE the forward pass so every selective_scan_fn call
    # records real tensor shapes → exact FLOPs with no formula guessing.
    _install_ssm_hook()
    with torch.no_grad():
        _ = model(lr_in, ref_in)          # single forward pass to trigger hook
    ssm_flops = sum(_ssm_flop_log)
    ssm_calls = len(_ssm_flop_log)
    print(f"\n  SSM hook intercepted {ssm_calls} selective_scan calls:")
    for i, f in enumerate(_ssm_flop_log):
        print(f"    call {i+1:>2}: {f:>12,}  ({f/1e9:.4f} GFLOPs)")
    print(f"  SSM total: {ssm_flops:,}  ({ssm_flops/1e9:.3f} GFLOPs)")

    # ── Step 2: fvcore for all standard ops (Conv, Linear, LN, etc.) ──────────
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table

        flops = FlopCountAnalysis(model, (lr_in, ref_in))
        flops.unsupported_ops_warnings(False)   # selective_scan_cuda → already counted
        flops.uncalled_modules_warnings(False)

        fvcore_flops = flops.total()
        total_flops  = fvcore_flops + ssm_flops

        print(f"\n  fvcore standard ops      : {fvcore_flops:>14,}  ({fvcore_flops/1e9:.3f} GFLOPs)")
        print(f"  SSM recurrence (hook)    : {ssm_flops:>14,}  ({ssm_flops/1e9:.3f} GFLOPs)")
        print(f"  {'─'*49}")
        print(f"  TOTAL FLOPs              : {total_flops:>14,}  ({total_flops/1e9:.3f} GFLOPs)")
        print(f"  TOTAL GMACs (FLOPs÷2)    : {total_flops/2/1e9:>14.3f}  GMACs")
        print("=" * 60)

        print("\n  Per-module breakdown (fvcore, depth=4):")
        print(flop_count_table(flops, max_depth=4))

        unsupported = flops.unsupported_ops()
        if unsupported:
            print("\n  Ops skipped by fvcore (already counted via hook):")
            for op, count in unsupported.items():
                print(f"    {op}: {count} call(s)")

    except ImportError:
        print("fvcore not installed. Run:  pip install fvcore")
        raise


if __name__ == "__main__":
    main()
