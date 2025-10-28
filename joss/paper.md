---
title: 'gwBOB: A Python Package for Analytical Merger-Ringdown Gravitational Waveforms'
tags:
  - Python
  - gravitational waves
  - general relativity
  - black holes
authors:
  - name: Anuj Kankani
    orcid: 0000-0002-7422-9137
    corresponding: true
    affiliation: "1, 2"
  - name: Angel Morales
    orcid: 0009-0000-4887-2172
    affiliation: "1, 2"
  - name: Suchindram Dasgupta
    orcid: 0009-0007-6658-0318
    affiliation: "1, 2"
  - name: Sean T. McWilliams 
    orcid: 0000-0003-2397-8290
    affiliation: "1, 2"


affiliations:
 - name: Department of Physics and Astronomy, West Virginia University, Morgantown, WV 26506, USA
   index: 1
 - name: Center for Gravitational Waves and Cosmology, West Virginia University, Morgantown, WV 26505, USA
   index: 2
date: 28 October 2025
bibliography: paper.bib
---

# Summary

gwBOB is an open-source Python package for generating gravitational waveforms for the merger and ringdown portion of black hole binary mergers using the Backwards-One-Body (BOB) model. The BOB model is an analytical, physically-motivated and highly accurate framework that models the merger-ringdown gravitational radiation based on the motion of null geodesics perturbed from the light ring of the remnant black hole [@mcwilliams2019analytical]. The model can be configured in various "flavors" based on the choice of initial conditions and the gravitational wave quantity being modeled. ``gwBOB`` provides a flexible and intuitive interface to generate waveforms from any flavor of BOB, use numerical relativity (NR) data for initial conditions, and validate BOB against NR waveforms. By providing a critical layer of abstraction over the BOB formalism, ``gwBOB`` greatly simplifies and streamlines the application of the model to various research problems.

# Background

The detection of gravitational waves (GW) has given us a new method to study the universe, separate but complementary to the detection of electromagnetic radiation. The most common sources detected are the inspiral, merger, and ringdown of two black holes. Extracting physical information from these faint signals, such as the masses, spins, and orbital parameters of the binary, requires the development of theoretical models for the gravitational radiation emitted during this process. The modeling process is particularly difficult for the merger portion of the coalescence due to the non-linear nature of Einstein's theory of General Relativity (GR). 

While NR simulations produce the most accurate waveforms, the significant computational expense of these simulations makes it impossible to generate enough waveforms to be used in detection pipelines. Instead, researchers must use semi-analytical waveform models, either calibrated to or directly interpolating NR simulations [@SEOBNR; @SEOBNRv5; @TEOB1; @TEOB2; @surr1; @surr2]. The non-linear nature of the merger results in this portion of the waveform being heavily reliant on NR information in all waveform models. As the sensitivity of current and future GW detectors increases, the limited coverage of NR catalogs across a higher dimensional parameter space may become a significant source of systematic error for these models. 

The Backwards-One-Body (BOB) model provides an analytical and physically motivated approach to merger-ringdown modeling that is minimally reliant on NR information. As [@kankani2025] shows, BOB can model the gravitational wave news, the first time derivative of the gravitational wave strain, to accuracy comparable to state of the art waveform models, all of which are heavily reliant on NR simulations. This package focuses on constructing BOB for the (s = -2, l = 2, m = 2) gravitational wave mode for quasi-circular and non-precessing configurations, but can be used for higher modes and precessing cases as well.

![Comparison of BOB and a NR waveform [@sxs_cat1] for the imaginary part of the (2, 2) mode of the News for a system with parameters similar to GW150914 [@gw150914]](BOB_news_0305.png)

# Statement of Need

`gwBOB` provides researchers with a robust and user-friendly implementation of BOB, eliminating the need for researchers to build custom implementations of BOB. The package allows researchers to easily configure and switch between different flavors of BOB, enabling them to choose the version best suited for their specific research problem. Its interface simplifies initialization by supporting both public NR catalogs [@sxs_cat1; @sxs_cat2; @sxs_cat3] and user provided NR data. For validation, the package includes built in utilities to streamline comparisons against NR and other semi-analytical waveform models. Furthermore, ``gwBOB`` includes additional routines for model comparison, such as one for inferring the remnant black hole's final mass and spin by minimizing the mismatch between a BOB waveform and a target waveform. This capability is particularly useful for quantitative comparisons with models constructed from a sum of quasinormal modes. By integrating utilities for initializing, configuring and validating the model, ``gwBOB`` provides the first complete open-source framework for incorporating any flavor of BOB into research workflows, dramatically reducing the complexity and time required for researchers to utilize the model.

# Related Work
``gwBOB`` is the first publicly available package that allows users to rapidly construct a variety of BOB flavors and validate them against NR waveforms. As part of the [nrpy](https://github.com/nrpy/nrpy) [@nrpy] code, a full inspiral-merger-ringdown model, using SEOBNRv5 [@SEOBNRv5] for the inspiral and a specific flavor of BOB for the merger-ringdown, is available. This package was extensively used in [@kankani2025] to extensively study BOB and provide comparisons to NR waveforms and semi-analytical waveform models. 

# Documentation

``gwBOB`` is distributed through PyPI and hosted on [GitHub](https://github.com/AnujKankani/BackwardsOneBody). Documentation is hosted on [readthedocs](https://backwardsonebody.readthedocs.io/en/latest/).

# Acknowledgements

AK and STM were supported in part by NSF CAREER grant PHY-1945130 and NASA grants 22-LPS22-0022 and 24-2024EPSCoR-0010. This research was made possible by the NASA West Virginia Space Grant Consortium, Grant \# 80NSSC20M0055.  The authors acknowledge the computational resources provided by the WVU Research Computing Thorny Flat HPC cluster, which is funded in part by NSF OAC-1726534.

# References


