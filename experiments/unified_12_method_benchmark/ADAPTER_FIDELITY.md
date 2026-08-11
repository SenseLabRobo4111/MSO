# Adapter fidelity and claim boundary

This experiment is an architecture-family benchmark under one occupancy-map
training and evaluation contract. Except where explicitly stated, an adapter is
not claimed to reproduce the complete loss, discriminator, schedule, data or
released performance of the named image-inpainting system.

The exact source revisions in `baseline_manifest.tsv` anchor the architectural
design and compatibility audit. The executable comparison implementations are
listed in `adapter_manifest.tsv` and hashed in every run manifest.

The benchmark therefore supports statements such as:

> Under a common occupancy objective and split, the MSO student is compared
> with ten adapted architecture families.

It does not support statements such as:

> MSO outperforms the original published systems under their official training
> protocols.

RePaint is retained in the registered roster but v1 does not convert its
iterative pretrained diffusion procedure into a single common supervised
adapter. Its row is reported as not executed instead of substituting an
incomparable checkpoint or a one-step surrogate.
