(Code in Development)
---
Please see more detailed documentation here (https://backwardsonebody.readthedocs.io/en/latest/index.html)!


## What is the Backwards One Body Model?

The **Backwards One Body (BOB) model** is a fully analytical approach to modeling gravitational waveforms from black hole binary mergers, as described in [arXiv:1810.00040](https://arxiv.org/abs/1810.00040). The BOB model is based on the physical insight that, during the late stages of binary evolution, the spacetime dynamics of the binary system closely resemble a linear perturbation of the final, stationary black hole remnant.

**Key features of the BOB model:**
- **Analytical accuracy:** Uses a physically motivated, closed form expression for the amplitude and frequency evolution.
- **Minimally Calibrated** Requires minimal calibration to numerical relativity (NR)
- **Minimal assumptions:** Assumes only that nonlinear effects remain small throughout the coalescence.
- **Physical foundation:** Relies on the tendency of the binary spacetime to behave like a perturbation of the merger remnant, reducing the need for phenomenological parameters.
- **Wide applicability:** Demonstrated to agree with state-of-the-art numerical relativity simulations across the quasi-circular and non-precessing parameter space.    

The BOB model provides a powerful, physically motivated, and computationally efficient tool for gravitational wave data analysis and theoretical studies.

---

## Features

- **Multiple waveform types:** Generate Psi4, News, and Strain waveforms
- **Flexible assumptions:** Choose between various initial conditions
- **Easy comparisons:** Easy comparisons to waveforms from the public SXS and CCE catalog, as well as raw NR data
- **Extensible and under active development**

---

## Installation

### Requirements

- (Windows users should use [WSL](https://docs.microsoft.com/en-us/windows/wsl/))
- [`kuibit`](https://github.com/SRombetto/kuibit)
- [`sxs`](https://github.com/sxs-collaboration/sxs)
- [`qnmfits`](https://github.com/sxs-collaboration/qnmfits) 
- [`jax`] (install the GPU compatible version if possible)
- [`scri`] (https://github.com/moble/scri)
- [`sympy`]
- [`numpy`] >2.0 (there is a kuibit incompatibility warning, numpy >2.0 does not cause issues.)
- [`scipy`]
- [`matplotlib`]


### Install via pip

