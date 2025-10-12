Welcome to gwBOB
=================

Backwards-One-Body Gravitational Waveform Package
-------------------------------------------------

**gwBOB** is a Python package for generating and analyzing gravitational waveforms
using the Backwards-One-Body model.

This documentation will help you:

- Install and get started!
- Explore the API for each module!
- Learn with examples!

What is the Backwards-One-Body Model?
-------------------------------------
The Backwards-One-Body (BOB) model is a analytical, physically motivated, and highly accurate model for the merger-ringdown radiation from black hole binary mergers. 
While a more detailed introduction to the physics and construction of BOB can be found in (https://arxiv.org/abs/1810.00040 and (paper in prep.)), we go over some of the basics here.

Calculating the motion of a bundle of null geodesics perturbed from the light ring of the remnant black hole, BOB models the amplitude of the merger-ringdown radiation as

.. math:: A = A_p\, \text{sech}\Big(\frac{t-t_p}{\tau}\Big)

where :math:`A_p` is the peak amplitude, :math:`t_p` is the peak time, and :math:`\tau` is the damping time of the fundamental quasinormal mode.

The frequency is then constructed through the relation 

.. math:: |\mathcal{N}|^2 \propto \Omega_{\text{orb}} \dot{\Omega}_{\text{orb}}

where :math:`\mathcal{N}` is the gravitational wave news

Quickstart
----------

.. toctree::
   :maxdepth: 2
   :caption: Quickstart

   quickstart

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: Modules

   gwBOB.BOB_utils
   gwBOB.gen_utils
   gwBOB.BOB_terms
   gwBOB.BOB_terms_jax
   gwBOB.ascii_funcs
   gwBOB.convert_to_strain_using_series
   gwBOB.mismatch_utils
   

Additional Pages
----------------

.. toctree::
   :maxdepth: 2

   faq
