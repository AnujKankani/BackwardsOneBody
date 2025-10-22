
<p align="center">
  <img src="docs/source/images/BOB_logo_v2.png" alt="gwBOB Logo" width="300">
</p>

<h1 align="center">gwBOB</h1>
<h3 align="center">The Backwards-One-Body Gravitational Waveform Package</h3>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/AnujKankani/BackwardsOneBody/actions/workflows/pytest.yml">
    <img src="https://github.com/AnujKankani/BackwardsOneBody/actions/workflows/pytest.yml/badge.svg" alt="Build Status">
  </a>
  <a href="https://backwardsonebody.readthedocs.io/en/latest/?badge=latest">
    <img src="https://readthedocs.org/projects/backwardsonebody/badge/?version=latest" alt="Documentation Status">
  </a>
  <a href="https://www.python.org/downloads/release/python-3120/">
    <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python Version">
  </a>
  <a href="https://github.com/AnujKankani/BackwardsOneBody">
    <img src="https://img.shields.io/badge/status-active-success.svg" alt="Project Status">
  </a>
</p>

## Getting Started
Please see more detailed documentation [here](https://backwardsonebody.readthedocs.io/en/latest/index.html)!\
And check out the tutorial notebooks in this repo!


## What is the Backwards One Body Model?

The **Backwards One Body (BOB) model** is an analytical and physically motivated approach to modeling gravitational waveforms from black hole binary mergers, as described in [arXiv:1810.00040](https://arxiv.org/abs/1810.00040). The BOB model is based on the physical insight that, during the late stages of binary evolution, the spacetime dynamics of the binary system closely resemble a linear perturbation of the final, stationary black hole remnant.

---

## Features
- **Analytical accuracy:**  Closed form expressions for the amplitude and frequency evolution.
- **Minimally Calibrated:** Requires minimal calibration to numerical relativity (NR)
- **Test all BOB flavors** Easily generate and switch between different "flavors" of BOB depending on your research problem.
- **Easy initialization** Easy initialization using SXS, CCE, or raw NR data. 
- **Beyond Kerr waveforms** Compare NR data to BOB waveforms generated with custom QNMs.
- **Easy comparisons:** Easy comparisons to waveforms from the public SXS and CCE catalog, as well as raw NR data.
- **Well Documented and Actively Developed**

<p align="center">
  <b>Generate plots like these with just a few lines of code!</b>
</p>
<p align="center">
  <img src="docs/source/images/BOB_news_0305.png">
</p>


## Requirements

- (Windows users should use [WSL](https://docs.microsoft.com/en-us/windows/wsl/))
- [`kuibit`](https://github.com/SRombetto/kuibit)
- [`sxs`](https://github.com/sxs-collaboration/sxs)
- [`qnmfits`](https://github.com/sxs-collaboration/qnmfits) 
- [`scri`](https://github.com/moble/scri)
- `jax` (install the GPU compatible version if possible)
- `sympy`
- `numpy`
- `scipy`
- `matplotlib`


## Install via pip

```bash
pip install gwBOB
```


## Citing this Code

If you use this code please cite
```text
@article{mcwilliams2019analytical,
  title={Analytical black-hole binary merger waveforms},
  author={McWilliams, Sean T},
  journal={Physical review letters},
  volume={122},
  number={19},
  pages={191102},
  year={2019},
  publisher={APS}
}
```

BOB paper to be added.

JOSS paper to be added.

## Contributing

Contributions are always welcome! If you find an issue, or have any questions on how to use the code, please raise an [issue](https://github.com/AnujKankani/BackwardsOneBody/issues) on this repo. If you want to contribute directly to the code, please fork the code and create a pull request!
