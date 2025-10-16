FAQ
=================

Code Conventions
-------------------

- omega vs Omega (phi vs Phi)

   - It is important to differentiate between the waveform frequency, denoted with a lowercase initial letter, and the orbital frequencies, denoted with an uppercase initial letter. 
     In our code the two values can be related by omega = m*Omega, where m is the mode number (and similarly phi = m*Phi). Please note the JAX compatible functions return little omega and phi, unlike the rest of the code.

- Sign conventions

   - The code assumes that if the final dimensionless spin has a positive value, then the remnant spin is aligned with the direction of the initial orbital angular momentum. If the spin is negative, that means a "spin flip" has taken place and the final spin is pointing opposite to the direction of the initial orbital angular momentum.

Potential Pitfalls
--------------------

- Switching between different flavors of BOB

   - As much as possible, the code is designed so that for any given SXS/CCE/NR case, the data only needs to be initialized once and the user can switch between different flavors of BOB without reinitializing the data. 
     This has several tradeoffs. First, the initialization process takes a little bit longer, as we store psi4, news, and strain data. However, once the initialization process is done, it is trivial to switch between different waveform quantities. 
     While switching from flavors that use t0 = -inf to finite t0 values should be seamless, switching from flavors that use t0 = -inf to finite t0 values requires the BOB.minf_t0 to be set to False. Furthermore, after each construction of BOB, parameters such as optimize_Omega0 retain their values, so the user should make sure to change these as appropriate.


Eccentric/Precessing/Higher modes
-------------------------------------

- This code has only been validated for the (2,2) mode of quasicircular and non-precessing systems. BOB's performance beyond this scope is underway, and in principle the code can be used with generic systems. Hoever, the performance of BOB may not be perfect!
  BOB has only been extensively studied for the (2,2) mode of quasicircular and non-precessing systems (Kankani and McWilliams 2025).

- Please raise issues on the GitHub pages if you have any questions!
