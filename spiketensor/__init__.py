"""One model for extracellular spike waveforms, plus its visualization package.

Every spike is at most R point sources; each source picks a place, a shape, a lag and a
loudness:

    Yhat_s = sum_{r=1..R} a_r * g(. ; mu_{n_r}, sigma_{n_r}) * (S_{tau_r} psi_{q_r})^T

The spatial atoms are analytic kernels with learned centres and scales; the shape codebook
{psi_q} is shared across the recording. Assignment is matching pursuit with exact
coefficient refits and the codebook update is a closed-form block -- there is no encoder
network anywhere in the pipeline.

Models that were once separate modules are configurations of `unified.Config`:

    Config(R=1, shape="free",   orthonormal=True)                    rank-one template
    Config(R=4, shape="free",   orthonormal=True)                    multi-source, free
    Config(R=4, shape="onehot", max_shift=0,  orthonormal=True)      one-hot shapes
    Config(R=4, shape="onehot", max_shift=10, orthonormal=True)      + shift invariance
    Config(R=8, shape="onehot", max_shift=10, P=2, cone_deg=35)      + prototype prior

Orthonormality and the prototype prior are mutually exclusive by construction, and
Config.validate() says so rather than silently dropping one.

Motion correction is canonical spikeinterface dredge_ap only (rigid and nonrigid); the
project's earlier internal soft/hard solve is not distributed.

See README.md for the panels, the browser and the 3-D viewer.
"""
__version__ = "0.2.0"
