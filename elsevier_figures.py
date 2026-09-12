"""Regenerate the publication figures under the Elsevier profile.

Reads figure_state.pkl (produced from the checkpoints - nothing is retrained)
and redraws one figure at a time so each can be reviewed before the next.

Usage:  python elsevier_figures.py <figure-key>
        python elsevier_figures.py --list
"""
import os, sys, pickle, warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = r"D:\AI-Projects\COAR-Q1"
# Prefer the slimmed cache (~31 MB) over the full snapshot (~808 MB). It holds
# the same objects the drawing code reads; the rest was tokenised datasets and
# raw text splits that no figure touches.
_SLIM = os.path.join(PROJECT, "figure_state_slim.pkl")
STATE = _SLIM if os.path.exists(_SLIM) else os.path.join(PROJECT, "figure_state.pkl")
OUTDIR = os.path.join(PROJECT, "outputs", "figures_elsevier")
STYLE_DIR = os.path.join(PROJECT, "figure_tools", "elsevier-figure-style",
                         "skills", "elsevier-figure-style", "scripts")
sys.path.insert(0, STYLE_DIR)

import elsevier_plot_style as eps

os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------- style
SPEC = eps.apply_elsevier_style()          # Arial, 11/10 pt, 0.6 pt rules, no legend frame
W1 = eps.figure_width("single")            # 3.543 in
W15 = eps.figure_width("one-half")         # 5.512 in
W2 = eps.figure_width("double")            # 7.480 in

# Okabe-Ito, matching the profile's own cycle and the original notebook.
PAL = {"coar": "#0072B2", "loo": "#D55E00", "random": "#666666",
       "d1": "#0072B2", "d2": "#009E73", "accent": "#CC79A7",
       "ink": "#000000", "muted": "#666666", "grid": "#D9D9D9"}
D1_LABEL, D2_LABEL = "Amazon Polarity", "Yelp Polarity"

# Rules and ticks sit back from the data; text stays full black.
RULE = "#4D4D4D"
TINT = {"d1": "#B8D6EA", "d2": "#B3E0D2", "loo": "#F2C7AC", "random": "#D9D9D9"}


def load_state():
    with open(STATE, "rb") as f:
        S = pickle.load(f)
    # budget_df / selectivity_ci_df are built in cells F11 and F12, which run
    # after the state snapshot is taken. Both have their own checkpoints.
    import joblib
    ck = os.path.join(PROJECT, "checkpoints")
    for stage in ("f11_budget", "f12_ci"):
        path = os.path.join(ck, stage + ".pkl")
        if os.path.exists(path):
            S.update(joblib.load(path))
    return S


def axis(ax, xlabel=None, ylabel=None, title=None, locator=True, frame="open"):
    """Profile styling (ticks, spines, grid policy) plus our labels.

    frame="open" drops the top and right spines and lightens the two that
    remain. A closed box draws a rectangle the data never uses; an open frame
    puts the ink on the axes the reader actually reads against.
    """
    eps.style_axis(ax, SPEC, set_major_locator=locator)
    if frame == "open":
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(RULE)
            ax.spines[side].set_linewidth(0.7)
        ax.tick_params(colors=RULE, labelcolor=PAL["ink"])
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return ax


def header(ax, letter, title=None, dy=7):
    """Panel letter and title as one left-aligned line above the axes.

    A centred bold title with the letter floating off to the left reads as two
    unrelated objects. Setting both flush to the axes' left edge, letter bold
    and title regular, gives the panel a single anchored headline.
    """
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction",
                xytext=(0, dy), textcoords="offset points",
                fontsize=10, fontweight="bold", va="bottom", ha="left",
                color=PAL["ink"], annotation_clip=False)
    if title:
        ax.annotate(title, xy=(0, 1), xycoords="axes fraction",
                    xytext=(24, dy), textcoords="offset points",
                    fontsize=10, va="bottom", ha="left",
                    color=PAL["ink"], annotation_clip=False)


def trim_spines(ax, x=None, y=None):
    """Clip the axis rules to the range the data actually occupies.

    A spine that runs on past the last data point (into the margin reserved for
    end labels, say) draws a line that means nothing. set_bounds ends the rule
    where the data ends.
    """
    if x is not None:
        ax.spines["bottom"].set_bounds(*x)
    if y is not None:
        ax.spines["left"].set_bounds(*y)


def end_label(ax, x, y, text, colour, dx=6, dy=0):
    """Label a series at its own end instead of in a legend box.

    Removes the legend frame and the eye's trip between key and curve.
    """
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=9.5, color=colour, va="center", ha="left",
                annotation_clip=False)


def panel(ax, letter, dx=-30, dy=5):
    """Panel label at a fixed offset in POINTS from the axes' top-left corner.

    Offsets in axes fractions (the previous approach) scale with panel width, so
    a label on a narrow panel drifted visibly further out than one on a wide
    panel. Points keep every label the same distance from its corner.
    """
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction",
                xytext=(dx, dy), textcoords="offset points",
                fontsize=10, fontweight="bold", va="bottom", ha="left",
                color=PAL["ink"], annotation_clip=False)


def save(fig, name, artwork_type="combination"):
    """Profile-backed export.

    bbox_inches is deliberately NOT 'tight': tight cropping shrinks the canvas
    below the exact column width, and the export auditor rejects that (OFF-06).
    The profile's own kwargs preserve the declared width.

    Writes PDF and TIFF (Elsevier submission formats), a flattened RGB PNG for
    preview, and a pickle that opens in FigureForge for manual touch-ups.
    """
    from io import BytesIO
    from PIL import Image

    base = os.path.join(OUTDIR, name)
    kw = eps.savefig_kwargs(SPEC, artwork_type=artwork_type)
    written = eps.save_figure(fig, base, spec=SPEC, formats=("pdf", "tiff"),
                              artwork_type=artwork_type)

    # PNG preview, flattened to RGB so it carries no needless alpha channel (OFF-09).
    png = f"{base}.png"
    with BytesIO() as buf:
        fig.savefig(buf, format="png", **kw)
        buf.seek(0)
        with Image.open(buf) as im:
            rgba = im.convert("RGBA")
    flat = Image.new("RGB", rgba.size, SPEC["figure"]["facecolor"])
    flat.paste(rgba, mask=rgba.getchannel("A"))
    flat.save(png, dpi=(kw["dpi"], kw["dpi"]))
    flat.close()

    pkl = f"{base}.pickle"
    with open(pkl, "wb") as f:
        pickle.dump(fig, f)

    w_mm = fig.get_size_inches()[0] * 25.4
    for p in [str(p) for p in written] + [png, pkl]:
        print(f"    {os.path.basename(p):<42} {os.path.getsize(p)/1e3:>9.1f} KB")
    print(f"    canvas width {w_mm:.1f} mm at {kw['dpi']} dpi")


# ================================================================ Figure 2
def fig2_surrogate_validity(S):
    """Surrogate validity: predicted vs true counterfactual margin."""
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.5), constrained_layout=True)

    for ax, results, label, colour, letter in [
        (axes[0], S["COAR_RESULTS"], D1_LABEL, PAL["d1"], "a"),
        (axes[1], S["COAR_RESULTS_D2"], D2_LABEL, PAL["d2"], "b"),
    ]:
        y_true = np.concatenate([r["y_test"] for r in results])
        y_pred = np.concatenate([r["y_test_pred"] for r in results])
        mean_r = float(np.mean([r["Test r"] for r in results]))

        # 3,000 heavily overplotted points per panel: the profile's default marker
        # size and opacity render them as one solid mass, hiding the distribution.
        # figure-style: allow STYLE-01 - overplotted scatter needs small, faint markers
        ax.scatter(y_true, y_pred, s=4, alpha=0.16, color=colour, edgecolors="none", rasterized=True)

        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        pad = 0.05 * (hi - lo)
        # figure-style: allow STYLE-01 - y=x is a reference annotation, not a data series
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=PAL["ink"], linestyle="--", linewidth=0.9, zorder=3, label="Perfect prediction")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")

        # One number only. The mask count moved to the caption, where it belongs.
        # \mathit is explicit: the profile sets mathtext.default=regular, so a
        # bare $r$ would render upright and Elsevier wants variables italic.
        ax.text(0.04, 0.965, f"$\\mathit{{r}}$ = {mean_r:.3f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=10, color=PAL["ink"])

        axis(ax,
             xlabel="True counterfactual margin",
             ylabel="COAR-predicted margin" if letter == "a" else None,
             )
        header(ax, letter, label)

    axes[0].legend(loc="lower right", fontsize=10, handlelength=1.8,
                   borderpad=0.3, handletextpad=0.5)
    save(fig, "fig2_surrogate_validity")
    plt.close(fig)
    print("\n  Caption. Figure 2. COAR surrogate predictions against true counterfactual\n"
          "  margins on held-out ablation masks for (a) Amazon Polarity and (b) Yelp\n"
          "  Polarity. Each point is one of 10 examples x 300 held-out masks; r is the\n"
          "  mean per-example Pearson correlation; the dashed line is perfect prediction.")


# ================================================================ Figure 3
def paired_ci(diffs, conf=0.95):
    """Mean paired difference with a t-based confidence interval."""
    from scipy.stats import t as student_t
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    m = float(np.mean(d))
    if n < 2:
        return m, 0.0, 0.0
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    h = float(student_t.ppf(0.5 + conf / 2, n - 1) * se)
    return m, m - h, m + h


def fig3_coar_vs_loo(S):
    """COAR against exhaustive LOO-lite, with the paired difference."""
    SEED = S.get("SEED", 42)
    fig = plt.figure(figsize=(W2, 3.1), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05])

    for col, (results, label, letter) in enumerate([
        (S["LOO_RESULTS"], D1_LABEL, "a"),
        (S["LOO_RESULTS_D2"], D2_LABEL, "b"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        coar_vals = [x["COAR Test r"] for x in results]
        loo_vals = [x["LOO Test r"] for x in results]
        xs = np.arange(len(results))

        ax.bar(xs - 0.19, coar_vals, 0.36, color=PAL["coar"], label="COAR", zorder=2)
        ax.bar(xs + 0.19, loo_vals, 0.36, color=PAL["loo"], label="LOO-lite", zorder=2)

        # figure-style: allow STYLE-01 - method-mean rules, not data series
        ax.axhline(np.mean(coar_vals), color=PAL["coar"], linestyle=":", linewidth=1.0, zorder=3)
        # figure-style: allow STYLE-01 - method-mean rules, not data series
        ax.axhline(np.mean(loo_vals), color=PAL["loo"], linestyle=":", linewidth=1.0, zorder=3)

        ax.set_xticks(xs)
        ax.set_xticklabels([str(x["Example"]) for x in results])
        # r is bounded above by 1; 1.15 leaves just enough room for the legend.
        ax.set_ylim(0, 1.15)
        axis(ax, xlabel="Example",
             ylabel="Test Pearson $\\mathit{r}$" if letter == "a" else None,
             locator=False)
        header(ax, letter, label)
        if letter == "a":
            ax.legend(loc="upper left", fontsize=9, ncol=2, columnspacing=1.0,
                      handlelength=1.2, borderpad=0.2, handletextpad=0.5)

    # -- Panel (c): paired differences with 95% CI --------------
    ax = fig.add_subplot(gs[0, 2])
    diff_d1 = [x["COAR Test r"] - x["LOO Test r"] for x in S["LOO_RESULTS"]]
    diff_d2 = [x["COAR Test r"] - x["LOO Test r"] for x in S["LOO_RESULTS_D2"]]

    for i, (diffs, colour) in enumerate([(diff_d1, PAL["d1"]), (diff_d2, PAL["d2"])]):
        m, lo, hi = paired_ci(diffs)
        jitter = (np.random.RandomState(SEED + i).rand(len(diffs)) - 0.5) * 0.16
        # figure-style: allow STYLE-01 - faint jittered per-example points behind the summary marker
        ax.scatter(np.full(len(diffs), i) + jitter, diffs, s=16, alpha=0.45, color=colour, edgecolors="none", zorder=2)
        # Marker size, cap size and line width all come from the profile rather
        # than being chosen by hand.
        L = SPEC["line"]
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", color=colour, zorder=4,
                    markersize=L["marker_size"], capsize=eps.errorbar_capsize(SPEC),
                    capthick=L["linewidth"], linewidth=L["linewidth"])
        ax.text(i + 0.20, m, f"{m:+.4f}", ha="left", va="center",
                fontsize=9, color=colour)

    # figure-style: allow STYLE-01 - zero reference line, not a data series
    ax.axhline(0, color=PAL["ink"], linewidth=0.8, zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([D1_LABEL.split()[0], D2_LABEL.split()[0]])
    ax.set_xlim(-0.5, 1.85)
    # The "CI crosses zero -> equivalent" note lived below the axis at y=-0.30,
    # which stretched the whole canvas and left a band of dead space. It is a
    # statement about the result, so it belongs in the caption.
    axis(ax, ylabel="COAR $-$ LOO-lite ($\\Delta\\mathit{r}$)",
         locator=False)
    header(ax, "c", "Paired difference")

    save(fig, "fig3_coar_vs_loo", artwork_type="line")
    plt.close(fig)

    m1, l1, h1 = paired_ci(diff_d1)
    m2, l2, h2 = paired_ci(diff_d2)
    print(f"\n  {D1_LABEL}: mean dr = {m1:+.4f}, 95% CI [{l1:+.4f}, {h1:+.4f}], n = {len(diff_d1)}")
    print(f"  {D2_LABEL}: mean dr = {m2:+.4f}, 95% CI [{l2:+.4f}, {h2:+.4f}], n = {len(diff_d2)}")
    print("\n  Caption. Figure 3. Attribution quality of COAR against exhaustive LOO-lite.\n"
          "  (a, b) Per-example Pearson correlation on identical held-out masks; dotted\n"
          "  lines mark each method's mean. (c) Paired per-example differences with 95%\n"
          "  confidence intervals; both intervals cross zero, so the two methods are\n"
          "  statistically indistinguishable on these data.")


# ================================================================ Figure 4
RATIO_FLOOR = 0.005   # reference denominators floored so the ratio cannot diverge


def _selectivity_table(edit_df_in, dataset_label):
    import pandas as pd
    rows = []
    for rec in edit_df_in.to_dict("records"):
        name = rec["Strategy"]
        if name.startswith("t-stat"):
            fam = "COAR (t-stat)"
        elif name.startswith("LOO"):
            fam = "LOO-lite"
        else:
            fam = "Random"
        tgt = abs(float(rec["Target Δmargin"]))
        ref = max(abs(float(rec["Reference Δmargin"])), RATIO_FLOOR)
        rows.append({"Dataset": dataset_label, "Strategy": fam, "k": int(rec["k"]),
                     "Target |d|": tgt, "Reference |d|": ref,
                     "Selectivity ratio": tgt / ref})
    return pd.DataFrame(rows)


def fig4_editing_selectivity(S):
    """Headline figure: selectivity ratio of each component-selection strategy."""
    import pandas as pd
    sel_d1 = _selectivity_table(S["edit_df"], D1_LABEL)
    sel_d2 = _selectivity_table(S["edit_df_d2"], D2_LABEL)
    sel_all = pd.concat([sel_d1, sel_d2], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.4), sharey=True,
                             constrained_layout=True)
    ks = sorted(sel_all["k"].unique())
    families = ["Random", "LOO-lite", "COAR (t-stat)"]   # COAR last so it sits on top
    fam_colour = {"Random": PAL["random"], "LOO-lite": PAL["loo"],
                  "COAR (t-stat)": PAL["coar"]}
    width = 0.26

    for ax, sel, label, letter in [
        (axes[0], sel_d1, D1_LABEL, "a"),
        (axes[1], sel_d2, D2_LABEL, "b"),
    ]:
        xs = np.arange(len(ks))
        for j, fam in enumerate(families):
            vals = [float(sel[(sel["k"] == k) & (sel["Strategy"] == fam)]
                          ["Selectivity ratio"].iloc[0]) for k in ks]
            # figure-style: allow STYLE-01 - hairline separators between grouped bars
            bars = ax.bar(xs + (j - 1) * width, vals, width, color=fam_colour[fam], label=fam if letter == "a" else None, zorder=2, edgecolor="white", linewidth=0.6)
            if fam == "COAR (t-stat)":
                for b, v in zip(bars, vals):
                    ax.text(b.get_x() + b.get_width() / 2, v * 1.10, f"{v:.0f}×",
                            ha="center", va="bottom", fontsize=9, color=PAL["coar"])

        # The horizontal break-even rule at ratio = 1 was removed on request: it
        # cut straight across both panels and read as clutter. The threshold is
        # still legible from the 10^0 tick, and the caption states it.

        ax.set_yscale("log")
        # Explicit limits: matplotlib's auto-padding on a log axis left a wide
        # empty band below the data.
        ax.set_ylim(0.01, 70)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"$\\mathit{{k}}$={k}" for k in ks])
        axis(ax, xlabel="Components ablated",
             ylabel="Selectivity ratio  |$\\Delta$target| / |$\\Delta$reference|"
                    if letter == "a" else None,
             locator=False)
        header(ax, letter, label)

    axes[0].legend(loc="upper left", fontsize=9, borderpad=0.3,
                   handlelength=1.6, handletextpad=0.5, labelspacing=0.35)
    save(fig, "fig4_editing_selectivity", artwork_type="line")
    plt.close(fig)

    pivot = sel_all.pivot_table(index=["Dataset", "k"], columns="Strategy",
                                values="Selectivity ratio").round(2)
    print("\n" + pivot.to_string())
    # Verify the caption's claim against the data rather than asserting it.
    wins_loo = wins_rnd = total = 0
    for sel in (sel_d1, sel_d2):
        for k in ks:
            g = {r["Strategy"]: r["Selectivity ratio"]
                 for _, r in sel[sel["k"] == k].iterrows()}
            total += 1
            wins_loo += g["COAR (t-stat)"] > g["LOO-lite"]
            wins_rnd += g["COAR (t-stat)"] > g["Random"]
    rnd_lo = sel_all[sel_all["Strategy"] == "Random"]["Selectivity ratio"].min()
    rnd_hi = sel_all[sel_all["Strategy"] == "Random"]["Selectivity ratio"].max()
    print(f"\n  COAR > LOO-lite in {wins_loo}/{total} cells; COAR > Random in {wins_rnd}/{total} cells")

    print(f"\n  Caption. Figure 4. Editing selectivity, the ratio of margin damage inflicted\n"
          f"  on target examples to that on reference examples, for (a) Amazon Polarity and\n"
          f"  (b) Yelp Polarity (log scale; reference denominators floored at {RATIO_FLOOR}).\n"
          f"  A ratio above 1 indicates a selective edit. COAR is more\n"
          f"  selective than LOO-lite at every budget on both datasets ({wins_loo}/{total}), and more\n"
          f"  selective than random selection in {wins_rnd} of {total} cases; the exception is Yelp at k=5.\n"
          f"  Random selection is a single draw per cell and is correspondingly unstable\n"
          f"  (ratios span {rnd_lo:.2f} to {rnd_hi:.2f}).")


# ================================================================ Figure 5a
def fig5_attribution_landscape(S):
    """Where the attributed components live: profile, depth, granularity."""
    imp1, imp2 = S["component_importance_df"], S["component_importance_df_d2"]
    k_edit = S["CONFIG"]["final_edit_k"]
    n_comp = S["NUM_COMPONENTS"]

    fig = plt.figure(figsize=(W2, 4.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1])

    # -- Panel (a): sorted importance profile -------------------
    ax = fig.add_subplot(gs[0, :])
    for imp_df, label, colour in [(imp1, D1_LABEL, PAL["d1"]),
                                  (imp2, D2_LABEL, PAL["d2"])]:
        vals = np.sort(imp_df["Mean |weight|"].values)[::-1]
        ax.plot(np.arange(1, len(vals) + 1), vals, color=colour, label=label,
                linewidth=SPEC["line"]["linewidth"])

    ax.set_yscale("log")
    ax.set_xscale("log")
    # Extra right margin holds the direct series labels that replace the legend.
    ax.set_xlim(0.9, n_comp * 2.3)

    # The edited components are a region, not a boundary. Shading rank <= k
    # reads at a glance and gives the panel a visual anchor; the original
    # hairline rule was easy to miss. (It was also drawn twice, once per
    # dataset, because axvline sat inside the loop.)
    ax.axvspan(0.9, k_edit, color=PAL["muted"], alpha=0.13, zorder=0, linewidth=0)
    # Label sits low inside the band: the curves run along the top of it, so an
    # upper placement collided with the Yelp trace.
    ax.text(1.05, ax.get_ylim()[0] * 1.5, f"edited\ntop {k_edit}",
            fontsize=9, color=PAL["muted"], va="bottom", ha="left")
    # The original title, "Attribution is concentrated in a small subset of
    # components", is an interpretation of the data rather than a label. It is
    # now the first sentence of the caption.
    # "Mean |attribution weight|" was long enough to span the whole axis height
    # and collide with the (a) panel label. Shortened to match panel (b), which
    # also makes the two panels consistent; the caption says these are
    # attribution weights.
    axis(ax, xlabel=f"Component rank (of {n_comp})",
         ylabel="Mean |weight|", locator=False)
    # Each curve is named at its own tail, so the reader never looks away from
    # the data to decode a key.
    for imp_df, label, colour in [(imp1, D1_LABEL, PAL["d1"]),
                                  (imp2, D2_LABEL, PAL["d2"])]:
        vals = np.sort(imp_df["Mean |weight|"].values)[::-1]
        end_label(ax, len(vals), vals[-1], label, colour)
    # The rule stops at rank 1224; the space beyond it belongs to the labels.
    trim_spines(ax, x=(0.9, n_comp))
    header(ax, "a")

    # -- Panel (b): by transformer depth ------------------------
    ax = fig.add_subplot(gs[1, 0])
    layers = sorted(imp1["Layer (depth)"].unique())
    w = 0.38
    for j, (imp_df, colour) in enumerate([(imp1, PAL["d1"]), (imp2, PAL["d2"])]):
        means = [float(imp_df[imp_df["Layer (depth)"] == L]["Mean |weight|"].mean())
                 for L in layers]
        # figure-style: allow STYLE-01 - hairline separators between grouped bars
        ax.bar(np.array(layers) + (j - 0.5) * w, means, w, color=colour, zorder=2, edgecolor="white", linewidth=0.5)
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{L}" for L in layers])
    # "By depth" removed: the x-axis already reads "Transformer layer".
    axis(ax, xlabel="Transformer layer", ylabel="Mean |weight|", locator=False)
    header(ax, "b", "By transformer depth")

    # -- Panel (c): by component granularity --------------------
    ax = fig.add_subplot(gs[1, 1])
    types = list(imp1["Type"].unique())
    # Only two categories here, so full-width bars looked heavy next to panel
    # (b)'s six. Narrower bars pulled toward the centre match its rhythm.
    wc = 0.26
    for j, (imp_df, label, colour) in enumerate([(imp1, D1_LABEL, PAL["d1"]),
                                                 (imp2, D2_LABEL, PAL["d2"])]):
        means = [float(imp_df[imp_df["Type"] == T]["Mean |weight|"].mean()) for T in types]
        # figure-style: allow STYLE-01 - hairline separators between grouped bars
        ax.bar(np.arange(len(types)) + (j - 0.5) * wc, means, wc, color=colour, label=label, zorder=2, edgecolor="white", linewidth=0.5)
    ax.set_xticks(np.arange(len(types)))
    ax.set_xticklabels(["Attention head\n(BIG)" if "Attention" in T else "FFN group\n(SMALL)"
                        for T in types])
    ax.set_xlim(-0.55, len(types) - 0.45)
    # "By granularity" removed: the x-axis already reads "Component type".
    axis(ax, xlabel="Component type", ylabel="Mean |weight|", locator=False)
    header(ax, "c", "By granularity")

    save(fig, "fig5_attribution_landscape", artwork_type="line")
    plt.close(fig)

    for imp_df, label in [(imp1, D1_LABEL), (imp2, D2_LABEL)]:
        vals = np.sort(imp_df["Mean |weight|"].values)[::-1]
        tot = vals.sum()
        print(f"  {label}: top-10 = {100*vals[:10].sum()/tot:.1f}%, "
              f"top-50 = {100*vals[:50].sum()/tot:.1f}%, "
              f"top-100 = {100*vals[:100].sum()/tot:.1f}% of total attribution mass")
    print(f"\n  Caption. Figure 5. Attribution is concentrated in a small subset of the\n"
          f"  {n_comp} components. (a) Sorted magnitude profile on log-log axes; the dotted\n"
          f"  line marks the k={k_edit} editing cut-off. (b) Mean magnitude by transformer\n"
          f"  depth. (c) Mean magnitude by component granularity. Colours as in (a).")


# ================================================================ Figure 5b
def fig5_edit_mechanism(S):
    """The two quantities behind the selectivity ratio: target vs reference damage."""
    edit_df, edit_df_d2 = S["edit_df"], S["edit_df_d2"]
    ks = sorted(edit_df["k"].unique())
    fams = [("Random", "Random", PAL["random"]),
            ("LOO", "LOO-lite", PAL["loo"]),
            ("t-stat", "COAR (t-stat)", PAL["coar"])]
    wid = 0.26

    def rows_for(df_in, prefix, col):
        return [float(df_in[(df_in["k"] == k) &
                            (df_in["Strategy"].str.startswith(prefix))][col].iloc[0])
                for k in ks]

    fig, axes = plt.subplots(2, 2, figsize=(W2, 4.8), sharex="col", sharey=True,
                             constrained_layout=True)

    for row, (df_in, dlabel) in enumerate([(edit_df, D1_LABEL), (edit_df_d2, D2_LABEL)]):
        for col, (which, colname) in enumerate([("Target set", "Target Δmargin"),
                                                ("Reference set", "Reference Δmargin")]):
            ax = axes[row, col]
            xs = np.arange(len(ks))
            for j, (prefix, nice, colour) in enumerate(fams):
                vals = rows_for(df_in, prefix, colname)
                # figure-style: allow STYLE-01 - hairline separators between grouped bars
                ax.bar(xs + (j - 1) * wid, vals, wid, color=colour, label=nice if (row == 0 and col == 0) else None, zorder=2, edgecolor="white", linewidth=0.5)

            # Signed quantity, so the zero baseline is structural, not decoration.
            # figure-style: allow STYLE-01 - zero baseline for a signed quantity
            ax.axhline(0, color=RULE, linewidth=0.8, zorder=3)
            ax.set_xticks(xs)
            ax.set_xticklabels([r"$\mathit{k}$=" + str(k) for k in ks])
            ax.set_ylim(-6.0, 0.7)
            axis(ax,
                 ylabel=(dlabel + "\n" + r"$\Delta$ margin") if col == 0 else None,
                 xlabel="Components ablated" if row == 1 else None,
                 locator=False)
            # Column identity is carried by the top row; repeating it underneath
            # would say the same thing twice.
            header(ax, "abcd"[row * 2 + col], which if row == 0 else None)

    axes[0, 0].legend(loc="lower left", fontsize=9, handlelength=1.4,
                      handletextpad=0.5, labelspacing=0.35, borderpad=0.3)

    save(fig, "fig5_edit_mechanism", artwork_type="line")
    plt.close(fig)

    print()
    print(f"  {'Dataset':<16}{'Strategy':<16}{'mean |dTgt|':>13}{'mean |dRef|':>13}{'ratio':>8}")
    for df_in, dlabel in [(edit_df, D1_LABEL), (edit_df_d2, D2_LABEL)]:
        for prefix, nice, _ in fams:
            t = float(np.abs(rows_for(df_in, prefix, "Target Δmargin")).mean())
            r = float(np.abs(rows_for(df_in, prefix, "Reference Δmargin")).mean())
            print(f"  {dlabel:<16}{nice:<16}{t:>13.3f}{r:>13.3f}{t / max(r, 1e-9):>8.2f}")

    print("\n  Caption. Figure 5. Mean correct-class margin change on the target set (a, c)\n"
          "  and the disjoint reference set (b, d) after ablating k components chosen by\n"
          "  each strategy, for Amazon Polarity (a, b) and Yelp Polarity (c, d). All four\n"
          "  panels share a common scale. LOO-lite produces large changes on both sets,\n"
          "  damaging the reference set as much as the target; COAR produces small changes\n"
          "  concentrated on the target.")

# ================================================================ Figure 6a
def fig6_magnitude_vs_specificity(S):
    """Attribution magnitude against target specificity: two different rankings."""
    from scipy.stats import spearmanr

    component_df = S["component_df"]
    is_head = (component_df["Type"] == "Attention head").values
    k_edit = S["CONFIG"]["final_edit_k"]
    n_comp = S["NUM_COMPONENTS"]

    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.9), constrained_layout=True)

    for ax, imp_df, tstat, sel_ids, label, letter in [
        (axes[0], S["component_importance_df"], S["t_stat"],
         S["FINAL_EDIT_IDS"], D1_LABEL, "a"),
        (axes[1], S["component_importance_df_d2"], S["t_stat_d2"],
         S["FINAL_EDIT_IDS_D2"], D2_LABEL, "b"),
    ]:
        mag = imp_df["Mean |weight|"].values
        ts = np.asarray(tstat, dtype=float)
        sel = np.zeros(n_comp, dtype=bool)
        sel[np.asarray(sel_ids)] = True

        # figure-style: allow STYLE-01 - 1,152 overplotted background points need small faint markers
        ax.scatter(mag[~is_head & ~sel], ts[~is_head & ~sel], s=5, alpha=0.16, color=PAL["muted"], edgecolors="none", label="FFN group", rasterized=True)
        # figure-style: allow STYLE-01 - heads are a distinct series marked by shape and size
        ax.scatter(mag[is_head & ~sel], ts[is_head & ~sel], s=15, alpha=0.75, color=PAL["loo"], marker="^", edgecolors="none", label="Attention head")
        # figure-style: allow STYLE-01 - open rings highlight the selected components over the cloud
        ax.scatter(mag[sel], ts[sel], s=44, facecolors="none", edgecolors=PAL["coar"], linewidths=1.3, zorder=5, label="Selected for editing")

        thresh = float(np.sort(ts)[-k_edit])
        # figure-style: allow STYLE-01 - selection threshold marker, not a data series
        ax.axhline(thresh, color=PAL["coar"], linestyle="--", linewidth=1.0, zorder=4)

        lo, hi = float(ts.min()), float(ts.max())
        span = hi - lo
        ax.set_ylim(lo - 0.10 * span, hi + 0.26 * span)
        ax.set_xscale("log")

        # Sat below the line at the left, where a head marker landed on it in
        # panel (b). It now rides ON the line at the far left, masked by a white
        # patch that breaks the dash cleanly; the wording moved to the caption.
        ax.text(0.012, thresh, "$\\mathit{t}$ = " + f"{thresh:.2f}",
                transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=9, color=PAL["coar"], zorder=6,
                bbox=dict(boxstyle="square,pad=0.25", facecolor="white",
                          edgecolor="none"))

        # The punchline of the figure. The original wrapped this in a white
        # rounded box; on a clean ground the box was drawing more attention
        # than the sentence, so the rule is gone and the colour carries it.
        n_head_sel = int(is_head[sel].sum())
        n_ffn_sel = int(sel.sum()) - n_head_sel
        # Bottom-right sat on a low-specificity head in panel (a). The band
        # above the rings is empty in both panels.
        ax.text(0.98, 0.97,
                f"selected: {n_ffn_sel} FFN groups, {n_head_sel} heads",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color=PAL["coar"])

        axis(ax,
             xlabel="Mean |attribution weight|",
             ylabel="Target specificity ($\\mathit{t}$-statistic)" if letter == "a" else None,
             locator=False)
        header(ax, letter, label)

    # Shared legend below both panels, laid out by constrained_layout so it
    # cannot collide with the axes or be clipped at export.
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="outside lower center", ncol=3, fontsize=9,
               scatterpoints=1, handletextpad=0.6, columnspacing=2.0)

    save(fig, "fig6_magnitude_vs_specificity", artwork_type="combination")
    plt.close(fig)

    print()
    print(f"  {'Dataset':<16}{'heads in top-10 |w|':>21}{'heads selected':>16}"
          f"{'median |w| rank':>18}{'Spearman':>10}")
    for imp_df, tstat, sel_ids, label in [
        (S["component_importance_df"], S["t_stat"], S["FINAL_EDIT_IDS"], D1_LABEL),
        (S["component_importance_df_d2"], S["t_stat_d2"], S["FINAL_EDIT_IDS_D2"], D2_LABEL),
    ]:
        mag = imp_df["Mean |weight|"].values
        ts = np.asarray(tstat, dtype=float)
        sel = np.asarray(sel_ids)
        order = np.argsort(-mag)
        mag_rank = {int(cid): r + 1 for r, cid in enumerate(order)}
        print(f"  {label:<16}{int(is_head[order[:10]].sum()):>21}"
              f"{int(is_head[sel].sum()):>16}"
              f"{int(np.median([mag_rank[int(c)] for c in sel])):>18}"
              f"{float(spearmanr(mag, ts)[0]):>10.3f}")

    print(f"\n  Caption. Figure 6. Mean attribution magnitude against target specificity for\n"
          f"  all {n_comp} components, on (a) Amazon Polarity and (b) Yelp Polarity. Attention\n"
          f"  heads (triangles) carry the largest attribution magnitudes but low specificity;\n"
          f"  every component selected for editing (rings, above the dashed selection threshold) is a\n"
          f"  feed-forward neuron group of only intermediate magnitude. Global importance and\n"
          f"  target specificity are therefore different rankings.")


# ================================================================ Figure 6b
def fig6_performance_and_locality(S):
    """Task performance of all five models, and how little the edit moves it."""
    k_edit = S["CONFIG"]["final_edit_k"]

    fig = plt.figure(figsize=(W2, 3.7), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1])

    # -- Panel (a): five-model accuracy -------------------------
    ax = fig.add_subplot(gs[0, 0])
    short_names = ["LogReg", "Linear SVM", "SVD+RF", "DistilBERT", "DistilBERT\n+COAR edit"]
    xs = np.arange(len(short_names))
    w = 0.38

    for j, (df_in, label, colour) in enumerate([
        (S["final5_df"], D1_LABEL, PAL["d1"]),
        (S["final5_df_d2"], D2_LABEL, PAL["d2"]),
    ]):
        accs = df_in["Accuracy"].astype(float).values
        # figure-style: allow STYLE-01 - hairline separators between grouped bars
        bars = ax.bar(xs + (j - 0.5) * w, accs, w, color=colour, label=label, zorder=2, edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, accs):
            # 7.5 pt: at 8 pt the DistilBERT pair's labels nearly touched. Still
            # above Elsevier's ~7 pt floor for normal text, and subordinate to
            # body size so the numbers do not compete with the bars.
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7.5, color=PAL["ink"])

    ax.set_xticks(xs)
    ax.set_xticklabels(short_names)
    # Truncated on purpose so the models can be told apart; stated in the caption
    # and the reason every bar carries its value.
    ax.set_ylim(0.75, 1.0)
    axis(ax, ylabel="Accuracy", locator=False)
    header(ax, "a", "Task performance")

    # -- Panel (b): what the edit moved -------------------------
    ax = fig.add_subplot(gs[0, 1])
    metrics = ["Accuracy", "F1", "ROC-AUC"]
    keys = ["accuracy", "f1", "roc_auc"]
    xs2 = np.arange(len(metrics))

    for j, (orig, edited, label, colour) in enumerate([
        (S["final_bert"], S["edited_bert"], D1_LABEL, PAL["d1"]),
        (S["final_bert_d2"], S["edited_bert_d2"], D2_LABEL, PAL["d2"]),
    ]):
        # Scaled by 1000 and declared in the axis label. Matplotlib's own
        # offset text ("1e-3" floating above the axis) is easy to miss, and
        # missing it makes a 0.0016 change look like 1.6.
        deltas = [(edited[k] - orig[k]) * 1e3 for k in keys]
        # figure-style: allow STYLE-01 - hairline separators between grouped bars
        ax.bar(xs2 + (j - 0.5) * w, deltas, w, color=colour, label=label, zorder=2, edgecolor="white", linewidth=0.6)

    # figure-style: allow STYLE-01 - zero baseline for a signed quantity
    ax.axhline(0, color=RULE, linewidth=0.8, zorder=3)
    ax.set_xticks(xs2)
    ax.set_xticklabels(metrics)
    axis(ax, ylabel="Change caused by the edit  ($\\times 10^{-3}$)", locator=False)
    header(ax, "b", "Edit locality")

    # One legend for both panels, below them. In the original it sat inside
    # panel (a) at lower right, directly on top of the DistilBERT bars.
    handles, labels_ = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="outside lower center", ncol=2, fontsize=9,
               handlelength=1.4, handletextpad=0.6, columnspacing=2.5)

    save(fig, "fig6_performance_and_locality", artwork_type="line")
    plt.close(fig)

    print()
    for orig, edited, label in [
        (S["final_bert"], S["edited_bert"], D1_LABEL),
        (S["final_bert_d2"], S["edited_bert_d2"], D2_LABEL),
    ]:
        parts = "  ".join(f"{m} {(edited[k] - orig[k]):+.4f}"
                          for m, k in zip(metrics, keys))
        print(f"  {label:<16}{parts}")

    print(f"\n  Caption. Figure 6. (a) Accuracy of the three TF-IDF baselines, the fine-tuned\n"
          f"  DistilBERT, and the COAR-edited model on both datasets. The y-axis is truncated\n"
          f"  at 0.75 so the models can be distinguished; every bar is labelled with its\n"
          f"  value. (b) Change in each metric caused by ablating the {k_edit} COAR-selected\n"
          f"  components, in units of 10^-3. The edit leaves original-task performance\n"
          f"  essentially untouched, which is what makes it an edit rather than damage.")


# ================================================================ Figure 7
def fig7_mask_budget(S):
    """Surrogate fidelity as a function of the compute budget used to fit it."""
    budget_df = S["budget_df"]
    n_comp = S["NUM_COMPONENTS"]

    # One panel, so the profile's single-panel rule applies: no title, no panel
    # letter. The claim moves to the caption. Sized to the one-half column.
    fig, ax = plt.subplots(figsize=(W15, 3.4), constrained_layout=True)

    n_ex = {}
    for label, colour in [(D1_LABEL, PAL["d1"]), (D2_LABEL, PAL["d2"])]:
        sub = budget_df[budget_df["Dataset"] == label]
        n_ex[label] = int(sub["Example"].nunique())
        g = sub.groupby("Forward passes")["Test r"]
        m, sd = g.mean(), g.std().fillna(0.0)
        ax.plot(m.index, m.values, "o-", color=colour, label=label,
                markersize=SPEC["line"]["marker_size"],
                linewidth=SPEC["line"]["linewidth"])
        # figure-style: allow STYLE-01 - spread band, deliberately faint behind the means
        ax.fill_between(m.index, m - sd, m + sd, color=colour, alpha=0.13, linewidth=0, zorder=0)

    # The claim is about a RANGE of budgets: everything up to 1224 passes costs
    # less than exhaustive leave-one-out. A shaded zone said this but laid a
    # colour field over half the data and muddied the spread bands where they
    # crossed it. A bracket along the top states the same span and leaves the
    # plotting area clean.
    ax.set_xlim(440, 2060)
    ax.annotate("", xy=(440, 0.975), xytext=(n_comp, 0.975),
                xycoords=("data", "axes fraction"),
                textcoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle="|-|", color=PAL["loo"],
                                linewidth=0.9, shrinkA=0, shrinkB=0,
                                mutation_scale=4))
    ax.text((440 + n_comp) / 2, 0.945,
            "cheaper than exhaustive LOO" + chr(10) + f"({n_comp} passes)",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=9, color=PAL["loo"], linespacing=1.3)

    axis(ax, xlabel="Forward passes used to fit the surrogate",
         ylabel="Test Pearson $\\mathit{r}$  (held-out masks)", locator=False)
    ax.legend(loc="lower right", fontsize=9, handlelength=1.8,
              handletextpad=0.6, borderpad=0.4)

    save(fig, "fig7_mask_budget", artwork_type="line")
    plt.close(fig)

    print()
    for label in (D1_LABEL, D2_LABEL):
        sub = budget_df[budget_df["Dataset"] == label]
        means = sub.groupby("Forward passes")["Test r"].mean()
        under = means[means.index <= n_comp]
        if len(under):
            at_loo, full = float(under.iloc[-1]), float(means.iloc[-1])
            print(f"  {label:<16} r = {at_loo:.4f} at {int(under.index[-1])} passes "
                  f"(<= LOO's {n_comp}), vs {full:.4f} at {int(means.index[-1])} "
                  f"-- retains {100 * at_loo / full:.1f}%")

    ns = " and ".join(str(v) for v in dict.fromkeys(n_ex.values()))
    print(f"\n  Caption. Figure 7. Surrogate fidelity on held-out masks as a function of the\n"
          f"  forward passes used to fit it, averaged over {ns} examples per dataset (shading:\n"
          f"  1 s.d.). The dashed line marks the cost of exhaustive leave-one-out ablation on\n"
          f"  this architecture ({n_comp} passes). Fidelity rises monotonically with budget and\n"
          f"  has not saturated at the largest budget tested.")


FIGURES = {
    "fig2": fig2_surrogate_validity,
    "fig3": fig3_coar_vs_loo,
    "fig4": fig4_editing_selectivity,
    "fig5a": fig5_attribution_landscape,
    "fig5b": fig5_edit_mechanism,
    "fig6a": fig6_magnitude_vs_specificity,
    "fig6b": fig6_performance_and_locality,
    "fig7": fig7_mask_budget,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("--list", "-l"):
        print("available:", ", ".join(FIGURES))
        return
    key = sys.argv[1]
    if key not in FIGURES:
        print(f"unknown figure {key!r}; available: {', '.join(FIGURES)}")
        sys.exit(1)
    print(f"loading state ...", flush=True)
    S = load_state()
    print(f"drawing {key} under the Elsevier profile")
    print(f"  canvas: double column = {W2:.3f} in, font = {SPEC['font']['sans_serif'][0]}")
    FIGURES[key](S)
    print(f"\n  -> {OUTDIR}")


if __name__ == "__main__":
    main()
