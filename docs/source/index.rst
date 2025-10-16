Welcome to gwBOB
=================

A Backwards-One-Body Gravitational Waveform Package
-------------------------------------------------------------------------

Features of gwBOB
-----------------

- Generate any flavor of BOB (psi4, news, strain, mass and current quadrupoles) using various initial conditions!
- Easy calling of SXS and CCE waveforms as well as user provided NR waveforms!
- Easy comparisons of BOB to NR waveforms as well as a sum of overtones!
- "Simple" parameter estimation using BOB!
- Open source, documented, and actively developed!

What is the Backwards-One-Body Model?
-------------------------------------
The Backwards-One-Body (BOB) model is a analytical, physically motivated, and highly accurate model for the merger-ringdown radiation from black hole binary mergers. 
While a more detailed introduction to the physics and construction of BOB can be found in (https://arxiv.org/abs/1810.00040 and (paper in prep.)), we go over some of the basics here.

Calculating the motion of a bundle of null geodesics perturbed from the light ring of the remnant black hole, BOB models the amplitude of the merger-ringdown radiation as

.. math:: A = A_p\, \text{sech}\Big(\frac{t-t_p}{\tau}\Big)

where :math:`A_p` is the peak amplitude, :math:`t_p` is the peak time, and :math:`\tau` is the damping time of the fundamental quasinormal mode.

The frequency is then constructed using the relation 

.. math:: |\mathcal{N}|^2 \propto \Omega_{\text{orb}} \dot{\Omega}_{\text{orb}}

where :math:`\mathcal{N}` is the gravitational wave news

While this appears simple enough, there are several caveats and important considerations. First, it is not obvious what gravitational wave quantity the amplitude evolution best represents.
Second the frequency evolution requires two initial conditions, :math:`t_0` and :math:`\Omega_0`. These two considerations result in a large variety of BOB "flavors" and is a primary reason for the development of this package.
By default we take the flavor of BOB used in (paper in prep.) which generates the gravitational wave news and takes :math:`t_0 = -\infty` and :math:`\Omega_0 \approx 0.155`.
This flavor generates the news to comparable accuracy of EOB and surrogate based waveform models while remaining minimally tuned to numerical relativity (NR). However, depending on the use-case, another flavor of BOB may be more appropriate. This package allows users to generate all 
reasonable flavors of BOB for easy comparison and analysis.

Of course, for most practical uses, we want the gravitational wave strain :math:`h(t)`. The fundamental BOB amplitude evolution does not directly model the strain accurately. 
Instead, BOB must be generated for either :math:`\Psi_4` or :math:`\mathcal{N}` and then integrated to obtain the strain. While we provide one analytical approximation based on a series expansion, 
this integration is largely left to the user as the "best" method (time or frequency domain numerical integration, Levin's method, Filon's method etc...) will be case dependent and several standard python libraries exist (numpy, scipy etc...) for this.

Lastly, this package is built for flexibility, convenience and using BOB as a stand-alone merger-ringdown model. Therefore, it is not highly optimized for speed nor is it integrated into a full inspiral-merger-ringdown model.
For a highly optimized IMR model that uses a EOB based inspiral and a BOB based merger-ringdown (using a specific flavor of BOB), please see (https://arxiv.org/abs/2508.20418 and https://github.com/nrpy/nrpy).


Quickstart
----------

.. toctree::
   :maxdepth: 2
   :caption: Quickstart

   quickstart
   faq

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: API

   apireference
