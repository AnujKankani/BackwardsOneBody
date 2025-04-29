# BackwardsOneBody

This code generates [BOB](https://arxiv.org/abs/1810.00040) waveforms. This code is still in development and may contain bugs.

Packages required:

kuibit

sxs

qnmfits (if using cce data NOTE: On windows this will require WSL). If you won't be loading in CCE data you do not need to install this package.

See detailed_documentation.py for examples on how to use the code.

Here is a full list of possible waveforms that can be generated.

Assuming BOB is for psi4
1. Psi4 with a finite t0 value. Omega_0 is calculated from the waveform
2. Psi4 taking t0 = -inf and Omega_0 = Omega_ISCO
3. Psi4 taking t0=-inf and Omega_0 is best fit
4. Strain obtained by building BOB for psi4 and taking |h| = |psi4|/(w_BOB^2)
   
Assuming BOB is for news
1. News with a finite t0 value. Omega_0 is calculated from the waveform
2. News taking t0 = -inf and Omega_0 = Omega_ISCO
3. News taking t0=-inf and Omega_0 is best fit
4. Strain obtained by building BOB for News and taking |h| = |N|/(w_BOB^2)
   
Assuming BOB is for strain
1. Strain with a finite t0 value. Omega_0 is calculated from the waveform
2. Strain taking t0 = -inf and Omega_0 = Omega_ISCO
3. Strain taking t0=-inf and Omega_0 is best fit

Additionaly, the phase alignment can be done at a specific time or this can also be best fit.
Notes:
1. The biggest problems occur when you use a finite t0 value. The corresponding Omega_0 value may result in imaginary frequencies, so this needs to be worked on a little bit.
2. No closed form solution was found for the phase when assuming BOB builds the news for finite t0 values, so the phase is integrated numerically.
2. 
