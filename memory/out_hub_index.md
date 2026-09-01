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
three sections:

- **Galleries** — directories containing an `index.html` (4 of them: 0018 and
  the three 0019 fractions). Each row joins the backing run directory
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
