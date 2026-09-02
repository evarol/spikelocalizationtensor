# Residuals-out hub index

**Created:** 2026-09-01
**Status:** working; regenerate any time

## Why the hub exists

The plot suites each produce their own `index.html` gallery (0018, the three
0019 fractions), but everything earlier — the 0014/0016/0017 loose-PNG suites,
`docs/`, `architecture/`, `plots/`, and the misc dirs — had no single entry
point. The SLT collision repo (`spiketensor/browser.py`) solves the same
problem for its fits with one self-contained HTML that embeds its inventory as
JSON; this hub copies that idea for the whole `residuals/out/` tree.

## What it is

`residuals/src/plots/build_out_index.py` (stdlib-only) walks `residuals/out/`
top level and writes `residuals/out/index.html`, a single dark-theme page with
three sections. **It auto-updates:** `build_plot_gallery.py` (used by every
0019+ plot-suite sbatch) and `kilosort_baseline_plots.py` (0022) both call
`build_out_index.build()` after writing their own gallery, so any suite job
that lands a gallery refreshes the hub in the same job. Suites set
`PYTHONPATH=$PWD/residuals/src:$PWD/residuals/src/plots`, which is what makes
the import resolve inside sbatch. A failed hub rebuild prints
"hub index rebuild skipped" and never fails the suite. Anything that adds
files *without* running a gallery builder (a lone script, an sbatch that only
drops PNGs) still needs the one-shot manual `python residuals/src/plots/build_out_index.py`.

Sections:

- **Galleries** — directories containing an `index.html` (9 of them: 0018, the
  three original 0019 fractions, the mean20/step20/kofn20/step0 sweep
  variants, and 0022 kilosort). Each row joins the backing run directory
  (`residuals/runs/dataset1_p1/<tag>/`) for `summary.json`/`config.json`
  metadata: n_events, n_rejected, detection threshold, the escalating
  all-channel bars (rendered like 0.05→0.25), stopping reason, PNG count,
  mtime, and links to both the gallery and the run data directory. The table
  is sortable by every column and filterable by family (0014…0019, misc).
- **Standalone figures** — top-level images (`0014_xyzsig_architecture.png`,
  `all_images.pptx`).
- **Collections** — every other directory, with a lazy thumbnail, a file
  count, and a collapsible list of all viewable files (482 links), one nested
  level deep for `plots/`' subdirectories. LaTeX build junk (.aux/.log/.out)
  is excluded.

Run it with the usual singularity wrapper (no GPU needed); it re-walks the
tree each time, so re-run after any new suite lands. A validation pass
confirmed every link target exists and the page JS (render, family filter,
column sort) executes cleanly; the node run also caught a real bug on the
first draft (non-PNG standalone figures crashed `addCard`).

## Caveats

- The page must stay self-contained for `file://` use: no fetch, everything
  embedded; the payload escapes `</` so `</script>` can't terminate early.
- Directory counts are top-level PNGs only; per-suite registered panel counts
  live in each gallery's own index page.
