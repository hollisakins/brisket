import numpyro
ncores = 4
numpyro.set_host_device_count(ncores)
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
# numpyro.set_host_device_count(4)

from jax import random, vmap
import jax.numpy as jnp
import numpy as np


rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

wav, flux, flux_err = np.loadtxt('/Users/hollis/Dropbox/research/NIRSpec/66964_spectrum.txt').T
flux *= 1e21
flux_err *= 1e21
flux_err = flux_err[(wav > 4)&(wav<5.31)]
flux = flux[(wav > 4)&(wav<5.31)]
wav = wav[(wav > 4)&(wav<5.31)]

import time
def compute_ymod(x, redshift, flux_broad, flux_narrow, fwhm_broad, fwhm_narrow, continuum=0.0):
    ymod = jnp.zeros_like(x)

    ymod += continuum

    mus = jnp.array([0.6563])
    for mu in mus:

        mu = mu * (1 + redshift)  # Apply redshift to the mean
        sigma_broad = mu * fwhm_broad /2.998e5 / 2.355
        sigma_narrow = mu * fwhm_narrow /2.998e5 / 2.355
        # Calculate the Gaussian profiles for broad and narrow components
        y_broad = 1/(sigma_broad*jnp.sqrt(2*jnp.pi)) * jnp.exp(-0.5 * ((x - mu) / sigma_broad)**2)
        y_broad *= flux_broad

        y_narrow = 1/(sigma_narrow*jnp.sqrt(2*jnp.pi)) * jnp.exp(-0.5 * ((x - mu) / sigma_narrow)**2)
        y_narrow *= flux_narrow

        # Combine the two components
        ymod += y_broad + y_narrow
    
    ymod2 = jnp.convolve(ymod, jnp.ones(300)/300, mode='same')  # Smooth the model
    # time.sleep(0.1)
    return ymod

import matplotlib.pyplot as plt
plt.rcParams['figure.constrained_layout.use'] = True
fig, ax = plt.subplots()
ax.step(wav, flux, where='mid', color='k')
ax.fill_between(wav, flux - flux_err, flux + flux_err, facecolor='k', edgecolor='none', alpha=0.2, step='mid')
ax.plot(wav, compute_ymod(wav, 7.037, 0.3, 0.3, 100, 1000, 1), color='C0', alpha=0.5, label='Initial Model')
plt.show()


def model(x, yerr, y=None):
    redshift = numpyro.sample("redshift", dist.Uniform(6.8, 7.5))
    flux_broad = numpyro.sample("flux_broad", dist.Uniform(0, 10))
    flux_narrow = numpyro.sample("flux_narrow", dist.Uniform(0, 10))
    fwhm_broad = numpyro.sample("fwhm_broad", dist.Uniform(700, 5000))
    fwhm_narrow = numpyro.sample("fwhm_narrow", dist.Uniform(100, 700))
    continuum = numpyro.sample("continuum", dist.Uniform(0, 10))

    ymod = compute_ymod(x, redshift, flux_broad, flux_narrow, fwhm_broad, fwhm_narrow, continuum)
    numpyro.sample("y", dist.Normal(ymod, yerr), obs=y)


# Run NUTS
mcmc = MCMC(
    sampler=NUTS(model), 
    num_warmup=1000, 
    num_samples=1000, 
    num_chains=1, 
    chain_method='vectorized', 
    jit_model_args=True)
mcmc.run(
    rng_key_, wav, flux_err, y=flux,
)
mcmc.print_summary()


import arviz as az
data = az.from_numpyro(mcmc)
az.plot_trace(data, show=True)

fig, ax = plt.subplots()
ax.step(wav, flux, where='mid', color='k')
ax.fill_between(wav, flux - flux_err, flux + flux_err, facecolor='k', edgecolor='none', alpha=0.2, step='mid')
samples = mcmc.get_samples()
for i in range(len(samples['redshift'])):
    redshift = samples['redshift'][i]
    flux_broad = samples['flux_broad'][i]
    flux_narrow = samples['flux_narrow'][i]
    fwhm_broad = samples['fwhm_broad'][i]
    fwhm_narrow = samples['fwhm_narrow'][i]
    continuum = samples['continuum'][i]
    ymod = compute_ymod(wav, redshift, flux_broad, flux_narrow, fwhm_broad, fwhm_narrow, continuum)
    ax.plot(wav, ymod, color='C0', alpha=0.05)
plt.show()
