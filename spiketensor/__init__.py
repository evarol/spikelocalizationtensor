"""Single-source tensor factorization of extracellular spike waveforms.

Each spike's (channel x time) waveform is reconstructed as one spatial footprint times one
time course:

    Y[s, c, t] ~ g_s(c) * (v_s @ a)[t]

where g_s is picked -- as a single discrete choice -- from a codebook of (lattice site x
radial profile) candidates, `a` is a low-dimensional time basis shared by every spike, and
v_s is a free per-spike coefficient vector. No gradient descent and no neural network: both
blocks of the fit have closed forms, so the objective decreases monotonically.

See README.md for the model, the solver, and what the visualizer panels show.
"""
__version__ = "0.1.0"
