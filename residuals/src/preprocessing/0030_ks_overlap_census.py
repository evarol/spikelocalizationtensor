"""Match a Kilosort (iblsorter ALF) spike train against a residual-pursuit run.

Both sides record events in raw sample-index units at the recording's own
sample rate (residual-pursuit's spike_times.npy, iblsorter's ALF
spikes.samples.npy), so matching is a direct integer-sample tolerance window
with no unit conversion -- the same design as the dataset1_p1 census
(session-022): time-only (any channel) and same-channel matching, at +/-0.5 ms,
against accepted events alone and against accepted+rejected proposals.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_ks(alf_dir):
    alf_dir = Path(alf_dir)
    samples = np.load(alf_dir / "spikes.samples.npy").astype(np.int64)
    clusters = np.load(alf_dir / "spikes.clusters.npy").astype(np.int64)
    cluster_channels = np.load(alf_dir / "clusters.channels.npy").astype(np.int64)
    raw_ind = np.load(alf_dir / "channels.rawInd.npy").astype(np.int64)
    raw_channel = raw_ind[cluster_channels[clusters]]

    good_clusters = set()
    with open(alf_dir / "cluster_KSLabel.tsv") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["KSLabel"].strip().lower() == "good":
                good_clusters.add(int(row["cluster_id"]))
    good_mask = np.isin(clusters, sorted(good_clusters))
    return samples, raw_channel, good_mask


def load_residual(run_dir):
    run_dir = Path(run_dir)
    accepted_t = np.load(run_dir / "spike_times.npy").astype(np.int64)
    accepted_c = np.load(run_dir / "spike_channels.npy").astype(np.int64)
    rejected_t = np.load(run_dir / "rejected_spike_times.npy").astype(np.int64)
    rejected_c = np.load(run_dir / "rejected_spike_channels.npy").astype(np.int64)
    return accepted_t, accepted_c, rejected_t, rejected_c


def time_only_coverage(query_times, reference_times, tol_samples):
    """Fraction of query_times within tol_samples of some reference time."""
    if query_times.size == 0:
        return 1.0, np.zeros(0, dtype=bool)
    ref_sorted = np.sort(reference_times)
    idx = np.searchsorted(ref_sorted, query_times)
    idx = np.clip(idx, 1, len(ref_sorted) - 1) if len(ref_sorted) else idx
    if len(ref_sorted) == 0:
        return 0.0, np.zeros(query_times.shape, dtype=bool)
    left = ref_sorted[np.clip(idx - 1, 0, len(ref_sorted) - 1)]
    right = ref_sorted[np.clip(idx, 0, len(ref_sorted) - 1)]
    hit = (np.abs(query_times - left) <= tol_samples) | (np.abs(query_times - right) <= tol_samples)
    return float(hit.mean()), hit


def same_channel_coverage(query_times, query_channels, reference_times, reference_channels, tol_samples):
    if query_times.size == 0:
        return 1.0, np.zeros(0, dtype=bool)
    hit = np.zeros(query_times.shape, dtype=bool)
    ref_channels = np.asarray(reference_channels)
    ref_times = np.asarray(reference_times)
    for ch in np.unique(query_channels):
        ref_mask = ref_channels == ch
        if not ref_mask.any():
            continue
        ref_sorted = np.sort(ref_times[ref_mask])
        q_mask = query_channels == ch
        q_times = query_times[q_mask]
        idx = np.searchsorted(ref_sorted, q_times)
        idx_lo = np.clip(idx - 1, 0, len(ref_sorted) - 1)
        idx_hi = np.clip(idx, 0, len(ref_sorted) - 1)
        ch_hit = (
            (np.abs(q_times - ref_sorted[idx_lo]) <= tol_samples)
            | (np.abs(q_times - ref_sorted[idx_hi]) <= tol_samples)
        )
        hit[q_mask] = ch_hit
    return float(hit.mean()), hit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ks-alf-dir", required=True, type=Path)
    parser.add_argument("--residual-run-dir", required=True, type=Path)
    parser.add_argument("--fs", type=float, required=True, help="recording sample rate (Hz)")
    parser.add_argument("--tolerance-ms", type=float, default=0.5)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    tol_samples = int(round(args.tolerance_ms / 1000.0 * args.fs))
    run_name = args.run_name or args.residual_run_dir.name

    ks_samples, ks_channel, ks_good = load_ks(args.ks_alf_dir)
    acc_t, acc_c, rej_t, rej_c = load_residual(args.residual_run_dir)
    prop_t = np.concatenate([acc_t, rej_t])
    prop_c = np.concatenate([acc_c, rej_c])

    result = {
        "run": run_name,
        "tolerance_ms": args.tolerance_ms,
        "fs": args.fs,
        "n_ks_spikes": int(ks_samples.size),
        "n_ks_good_spikes": int(ks_good.sum()),
        "n_accepted": int(acc_t.size),
        "n_rejected": int(rej_t.size),
    }

    # precision: fraction of accepted/proposed residual-pursuit events that
    # coincide with some Kilosort spike time (any channel).
    result["accepted_time_precision"], _ = time_only_coverage(acc_t, ks_samples, tol_samples)
    result["proposal_time_precision"], _ = time_only_coverage(prop_t, ks_samples, tol_samples)

    # recall: fraction of Kilosort spikes (all / good only) covered by
    # accepted events, and separately by accepted+rejected proposals.
    result["recall_all_ks_by_accepted"], _ = time_only_coverage(ks_samples, acc_t, tol_samples)
    result["recall_all_ks_by_proposals"], _ = time_only_coverage(ks_samples, prop_t, tol_samples)
    good_samples = ks_samples[ks_good]
    result["recall_good_ks_by_accepted"], _ = time_only_coverage(good_samples, acc_t, tol_samples)
    result["recall_good_ks_by_proposals"], _ = time_only_coverage(good_samples, prop_t, tol_samples)

    # same-channel variants (channel-assignment bookkeeping check).
    result["recall_all_ks_by_accepted_samechannel"], _ = same_channel_coverage(
        ks_samples, ks_channel, acc_t, acc_c, tol_samples
    )
    result["recall_good_ks_by_accepted_samechannel"], _ = same_channel_coverage(
        good_samples, ks_channel[ks_good], acc_t, acc_c, tol_samples
    )
    _, acc_hit_samechannel = same_channel_coverage(acc_t, acc_c, ks_samples, ks_channel, tol_samples)
    result["accepted_time_precision_samechannel"] = float(acc_hit_samechannel.mean()) if acc_t.size else 1.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{run_name}_census.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
