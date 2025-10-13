(Code in Development)
(Generated via Perplexity AI)
---

## 🌀 What is the Backwards One Body Model?

The **Backwards One Body (BOB) model** is a fully analytical approach to modeling gravitational waveforms from black hole binary mergers, as described in [arXiv:1810.00040](https://arxiv.org/abs/1810.00040). The BOB model is based on the physical insight that, during the late stages of binary evolution, the spacetime dynamics of the binary system closely resemble a linear perturbation of the final, stationary black hole remnant-even before the actual merger.

**Key features of the BOB model:**
- **Analytical accuracy:** Accurately models the late inspiral, merger, and ringdown phases for arbitrary mass ratios and spins, including higher harmonics.
- **Minimal assumptions:** Assumes only that nonlinear effects remain small throughout the coalescence.
- **Physical foundation:** Relies on the tendency of the binary spacetime to behave like a perturbation of the merger remnant, reducing the need for phenomenological parameters.
- **Wide applicability:** Demonstrated to agree with state-of-the-art numerical relativity simulations.

The BOB model provides a powerful, physically motivated, and computationally efficient tool for gravitational wave data analysis and theoretical studies.

---

## 🚀 Features

- **Multiple waveform types:** Generate Psi4, News, and Strain waveforms
- **Flexible assumptions:** Choose between finite or infinite $t_0$ and various $\Omega_0$ options (auto, ISCO, best-fit)
- **Phase alignment:** Align phases at a specific time or via best-fit
- **Data compatibility:** Works with SXS, CCE or psi4 data from NR simulations
- **Easy visualization:** Plot and compare waveforms
- **Extensible and under active development**

---

## 📦 Installation

### Requirements

- Python 3.7+
- [`kuibit`](https://github.com/SRombetto/kuibit)
- [`sxs`](https://github.com/sxs-collaboration/sxs)
- [`qnmfits`](https://github.com/sxs-collaboration/qnmfits) *(only needed for CCE data; Windows users need [WSL](https://docs.microsoft.com/en-us/windows/wsl/))*

### Install via pip

