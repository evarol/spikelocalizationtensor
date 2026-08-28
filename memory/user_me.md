# User Profile

## Working style
- Neuroscience research workflow using Python, SLURM, Singularity, SpikeGLX recordings, and GPU jobs.
- Prefers implementing and testing concrete ideas quickly, then inspecting failures from short queued jobs.
- Wants the analytic solver to remain centered on localization and reconstruction rather than unit-clustering as the endpoint.

## Communication preferences
- Be direct and stay at the requested abstraction level.
- Ask questions when a scientific choice materially changes the result.
- Do not spend time validating or adopting an external stack after the user has asked for an in-house implementation.
- Write displayed formulas with `$$ ... $$` when documenting solver mathematics.

## Must-haves
- Run all SLURM commands outside the sandbox.
- Record durable session findings in `memory/` using the `new-memory` workflow; do not update or recreate `RUNNING_LOG.md`.
- Use `spikeglx.Reader` directly for raw BIN/CBIN access.
- Do not use the `iblsorter` library in the new residual pipeline; its source is only an algorithm reference.
- Detect with full template deconvolution over valid time samples, using templates derived from the spatial and temporal codebooks.
- Use 48 µm channel-map neighborhoods around each anchor channel.
- Localize and reconstruct each accepted component, subtract it, and repeat on the residual.
- Preserve unrelated worktree files and keep commits coherent.
