"""
tb_hybrid_verify.py — TB-hybrid verification (nested-grid tail subtraction)
============================================================================

Moved out of tests/ on 2026-07-23 (P1): the full production scan used to live
in tests/test_graphene_tb_hybrid.py and ran a ~9 minute scan on *import*, with
no assertions.  It is now a standalone script run via

    python scripts/tb_hybrid_verify.py

and the pytest entry point is the fast, seconds-scale
tests/test_graphene_tb_hybrid.py (V0 + V1b only).

2026-07-23: Updated to use nested-grid approach.
Root cause fix: analytical chi0_und vs numerical Dirac(compact) discretization
mismatch was producing spurious positive Im[chi0] (~316% of DOS).
Fix: all-numerical Dirac tail with nested grids (compact subset of large),
ensuring exact cancellation in the compact region.
"""
import os, sys, time, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# Try repo import first, fall back to dump/
try:
    from src.validation.graphene_tb_hybrid import (
        build_model, chi0_tb_hybrid, chi0_to_physical,
        rpa_dielectric_phys, dos_dirac_model, kF_model,
        plasmon_freq_phys, VF_PHYS,
    )
except ImportError:
    sys.path.insert(0, "D:/erpt/dump")
    import graphene_tb_hybrid as gth_module
    build_model = gth_module.build_model
    chi0_tb_hybrid = gth_module.chi0_tb_hybrid
    chi0_to_physical = gth_module.chi0_to_physical
    rpa_dielectric_phys = gth_module.rpa_dielectric_phys
    dos_dirac_model = gth_module.dos_dirac_model
    kF_model = gth_module.kF_model
    plasmon_freq_phys = gth_module.plasmon_freq_phys
    VF_PHYS = gth_module.VF_PHYS
try:
    import graphene_rpa as gr
except ImportError:
    sys.path.insert(0, "D:/erpt/dump")
    import graphene_rpa as gr


def main():
    Ef, kappa, eta, beta = 2.0, 1.0, 0.005, 200.0
    model = build_model()
    trapz = np.trapezoid
    D_phys = dos_dirac_model(Ef) / 2.46**2
    kf_phys = kF_model(Ef) / 2.46

    results = []
    def report(name, ok, detail):
        results.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    # ── Static chi0 ──
    print("== V1 Static chi0 ==")
    t0 = time.time()
    q_st = np.linspace(0.01, 0.80, 30)
    chi_st = chi0_tb_hybrid(model, q_st, np.array([0.0]), Ef=Ef,
        Nk=200, Ntheta=120, k_cut_factor=2, eta=1e-4, verbose=False)
    chi_st_phys = chi0_to_physical(chi_st[0])
    q_st_phys = q_st / 2.46
    re_r = -chi_st_phys.real / D_phys
    dev = np.max(np.abs(re_r[q_st_phys < 1.5*kf_phys] - 1)) * 100
    # TB trigonal warping: (Ef/t)^2 ~ 50% at Ef=2eV, so ~35% is expected
    report("V1a TB plateau vs Dirac DOS", dev < 40,
           f"max dev = {dev:.1f}% (TB trigonal warping at Ef={Ef} eV)")
    report("V1b Static Im=0", np.max(np.abs(chi_st_phys.imag/D_phys))*100 < 3,
           f"max Im/D = {np.max(np.abs(chi_st_phys.imag/D_phys))*100:.2f}%")
    cf = gr.chi0_static_closedform(q_st_phys, Ef)
    dev_cf = np.max(np.abs(chi_st_phys.real/cf - 1)) * 100
    report("V1c vs Dirac closed form", dev_cf < 40,
           f"max dev = {dev_cf:.1f}%")
    print(f"  ({time.time()-t0:.1f}s)")

    # ── Production scan ──
    print("\n== Production scan ==")
    t0 = time.time()
    q_grid = np.linspace(0.015, 0.75, 20)
    w_grid = np.linspace(0.01, 5.0, 301)
    chi0 = chi0_tb_hybrid(model, q_grid, w_grid, Ef=Ef, Nk=200, Ntheta=120,
        k_cut_factor=2, eta=eta, verbose=True)
    chi0_phys = chi0_to_physical(chi0)
    q_phys = q_grid / 2.46
    eps, eps_inv, elf = rpa_dielectric_phys(chi0_phys, q_phys, kappa=kappa)
    print(f"  ({time.time()-t0:.1f}s)")

    # V0 Retardation
    v0 = np.max(chi0_phys.imag)
    report("V0 Im chi0 <= 0", v0 < 3e-2 * D_phys,
           f"max Im = {v0:.1e} (D={D_phys:.1e}, {v0/D_phys*100:.2f}%)")

    # V2 Plasmon
    wpl = np.zeros(len(q_grid))
    for iq in range(len(q_grid)):
        i0 = np.argmax(elf[2:, iq]) + 2
        if 0 < i0 < len(w_grid)-1:
            y0,y1,y2 = elf[i0-1,iq], elf[i0,iq], elf[i0+1,iq]
            denom = y0-2*y1+y2
            shift = 0.5*(y0-y2)/denom if abs(denom)>1e-30 else 0
            wpl[iq] = w_grid[i0] + np.clip(shift, -1, 1)*(w_grid[1]-w_grid[0])
        else:
            wpl[iq] = w_grid[i0]

    A_th = plasmon_freq_phys(1.0, Ef, kappa)
    mask_fit = (q_phys >= 0.02) & (q_phys <= 0.15) & (wpl > 0)
    if mask_fit.sum() >= 3:
        A_fit = wpl[mask_fit][0] / np.sqrt(q_phys[mask_fit][0])
        report("V2 sqrt(q) prefactor", abs(A_fit/A_th-1) < 0.15,
               f"A={A_fit:.2f} vs A_th={A_th:.2f} ({(A_fit/A_th-1)*100:+.1f}%)")

    # V3 f-sum
    fsum = trapz(w_grid[:,None]*elf, w_grid, axis=0)
    fsm = q_phys <= 0.12
    fr = fsum[fsm] / gr.fsum_th(q_phys[fsm], Ef, kappa=kappa)
    report("V3 f-sum", 0.6 < fr.min() and fr.max() < 1.6,
           f"[{fr.min():.3f}, {fr.max():.3f}]")

    # V4 Re eps zero-cross = ELF peak
    v4_devs = []
    for iq in range(len(q_grid)):
        if not mask_fit[iq]:
            continue
        re_eps = eps[:, iq].real
        sgn = np.sign(re_eps)
        cross = np.where(np.diff(sgn) > 0)[0]
        if len(cross) > 0:
            ic = cross[-1]
            dw = w_grid[1]-w_grid[0]
            w0 = w_grid[ic] + dw*(-re_eps[ic]/(re_eps[ic+1]-re_eps[ic]))
            v4_devs.append(abs(w0-wpl[iq])/wpl[iq])
    v4_max = max(v4_devs) if v4_devs else np.nan
    report("V4 Re eps zero=ELF", v4_max < 0.05,
           f"max rel dev = {v4_max*100:.2f}% (n={len(v4_devs)})")

    # ── Figures ──
    FIGDIR = "figures-graphene-tb-rpa"
    os.makedirs(FIGDIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7,5), dpi=150)
    Q, W = np.meshgrid(q_phys/kf_phys, w_grid)
    vm = np.percentile(elf, 99)
    pc = ax.pcolormesh(Q, W, elf, shading="auto", cmap="viridis", vmin=0, vmax=vm)
    fig.colorbar(pc, ax=ax, label="ELF")
    qq = np.linspace(q_phys[0], q_phys[-1], 200)
    ax.plot(qq/kf_phys, VF_PHYS*qq, "w--", lw=1, label="v_F q")
    ax.plot(qq/kf_phys, 2*Ef-VF_PHYS*qq, "w-.", lw=1, label="2E_F-v_F q")
    ax.plot(qq/kf_phys, plasmon_freq_phys(qq, Ef, kappa), "r-", lw=1.2, alpha=0.7)
    sel = wpl > 0
    ax.plot(q_phys[sel]/kf_phys, wpl[sel], "o", ms=3, color="red", mfc="none", label="ELF peak")
    ax.legend(fontsize=8); ax.set_xlabel("q/k_F"); ax.set_ylabel("omega [eV]")
    ax.set_title(f"TB-hybrid ELF (nested-grid): E_F={Ef} eV")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/elf-hybrid-nested.png"); plt.close()

    fig, ax = plt.subplots(figsize=(7,5), dpi=150)
    ax.plot(q_st_phys/kf_phys, -gr.chi0_static_closedform(q_st_phys, Ef)/D_phys,
            "-", color="gray", lw=1.5, label="Dirac")
    ax.plot(q_st_phys/kf_phys, re_r, "o", ms=4, color="C0", label="TB-hybrid")
    ax.axhline(1, color="gray", ls="--", lw=0.8)
    ax.axvline(1, color="red", ls=":", lw=1, label="q=2k_F")
    ax.legend(); ax.set_xlabel("q/k_F"); ax.set_ylabel("-chi0/D(E_F)")
    ax.set_title("Static chi0: TB-hybrid (nested-grid) vs Dirac")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/static-hybrid-nested.png"); plt.close()

    # Summary
    print("\n" + "="*50)
    n_pass = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(f"  => {n_pass}/{len(results)} PASS")


if __name__ == "__main__":
    main()
