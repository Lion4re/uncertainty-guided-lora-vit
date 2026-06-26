# License Notice

This repository contains thesis-specific modifications and experiment utilities built on top of the
official PACE vision codebase:

https://github.com/MaxwellYaoNi/PACE

The upstream PACE repository is distributed under the MIT License. The MIT license text and upstream
copyright notice are included in `LICENSE`.

The thesis-specific additions include uncertainty-guided consistency losses, PAC-Bayes/IVON/Bayesian
LoRA diagnostics, shift/OOD evaluation, ensemble evaluation, plotting utilities, runners, configs,
tests, curated result artifacts, and documentation. Unless otherwise noted, these additions are
intended to be distributed under the same MIT terms as the upstream code.

This notice is included to make the provenance explicit: the adapter and PACE training
infrastructure follow the original PACE repository, while the uncertainty-guided, Bayesian,
diagnostic, and thesis reproducibility layers were added for this research artifact.
