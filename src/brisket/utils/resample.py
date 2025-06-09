'''
Adapted from the `spectres` package by Adam Carnall
'''
from __future__ import print_function, division, absolute_import
import warnings
import jax.numpy as jnp
from jax import lax

def make_bins(midpoints, fix_low=None, fix_high=None):
    """ A general function for turning an array of bin midpoints into an
    array of bin positions. Splits the distance between bin midpoints equally in linear space.

    Parameters
    ----------
    midpoints : numpy.ndarray
        Array of bin midpoint positions

    fix_low : float, optional
        If set, the left edge of the first bin will be fixed to this value

    fix_high : float, optional
        If set, the right edge of the last bin will be fixed to this value
    """

    bins = jnp.zeros(midpoints.shape[0]+1)
    if fix_low is not None:
        bins = bins.at[0].set(fix_low)
    else:
        bins = bins.at[0].set(midpoints[0] - (midpoints[1]-midpoints[0])/2)
    if fix_high is not None:
        bins = bins.at[-1].set(fix_high)
    else:
        bins = bins.at[-1].set(midpoints[-1] + (midpoints[-1]-midpoints[-2])/2)
    bins = bins.at[1:-1].set((midpoints[1:] + midpoints[:-1])/2)

    return bins

def resample_spectrum(new_wavs, spec_wavs, spec_fluxes, spec_errs=None, fill=0.0, verbose=True):
    """
    JAX-compatible function for resampling spectra (and optionally associated
    uncertainties) onto a new wavelength basis.
    Adapted from the `spectres` package by Adam Carnall.

    Parameters
    ----------
    new_wavs : jax.numpy.ndarray
        Array containing the new wavelength sampling desired for the
        spectrum or spectra.

    spec_wavs : jax.numpy.ndarray
        1D array containing the current wavelength sampling of the
        spectrum or spectra.

    spec_fluxes : jax.numpy.ndarray
        Array containing spectral fluxes at the wavelengths specified in
        spec_wavs, last dimension must correspond to the shape of
        spec_wavs. Extra dimensions before this may be used to include
        multiple spectra.

    spec_errs : jax.numpy.ndarray (optional)
        Array of the same shape as spec_fluxes containing uncertainties
        associated with each spectral flux value.

    fill : float (optional)
        Where new_wavs extends outside the wavelength range in spec_wavs
        this value will be used as a filler in new_fluxes and new_errs.

    verbose : bool (optional)
        Setting verbose to False will suppress the default warning about
        new_wavs extending outside spec_wavs and "fill" being used.

    Returns
    -------
    new_fluxes : jax.numpy.ndarray
        Array of resampled flux values, last dimension is the same
        length as new_wavs, other dimensions are the same as
        spec_fluxes.

    new_errs : jax.numpy.ndarray (optional)
        Array of uncertainties associated with fluxes in new_fluxes.
        Only returned if spec_errs was specified.
    """
    
    # Rename the input variables for clarity
    old_wavs = spec_wavs
    old_fluxes = spec_fluxes
    old_errs = spec_errs

    # Make arrays of edge positions and widths for the old and new bins
    old_edges = make_bins(old_wavs)
    new_edges = make_bins(new_wavs)
    old_widths = jnp.diff(old_edges)
    new_widths = jnp.diff(new_edges)

    # Generate output arrays to be populated
    new_fluxes_shape = old_fluxes.shape[:-1] + (len(new_wavs),)
    new_fluxes = jnp.zeros(new_fluxes_shape)

    if old_errs is not None:
        if old_errs.shape != old_fluxes.shape:
            raise ValueError("If specified, spec_errs must be the same shape "
                             "as spec_fluxes.")
        new_errs = jnp.zeros(new_fluxes_shape)
    else:
        new_errs = None

    def process_bin(j, arrays):
        new_fluxes, new_errs = arrays
        
        # Check if new bin extends outside old wavelength range
        outside_range = (new_edges[j] < old_edges[0]) | (new_edges[j+1] > old_edges[-1])
        
        def fill_bin():
            new_flux = jnp.full(old_fluxes.shape[:-1], fill)
            new_err = jnp.full(old_fluxes.shape[:-1], fill) if old_errs is not None else None
            return new_flux, new_err
        
        def process_normal_bin():
            # Find overlapping old bins using vectorized operations
            # Start bin: last bin where old_edges[i+1] <= new_edges[j]
            start_mask = old_edges[1:] <= new_edges[j]
            start = jnp.sum(start_mask.astype(jnp.int32))
            
            # Stop bin: last bin where old_edges[i+1] < new_edges[j+1] 
            stop_mask = old_edges[1:] < new_edges[j+1]
            stop = jnp.sum(stop_mask.astype(jnp.int32))
            
            # Case 1: New bin is fully inside one old bin
            def single_bin_case():
                flux = old_fluxes[..., start]
                err = old_errs[..., start] if old_errs is not None else None
                return flux, err
            
            # Case 2: New bin spans multiple old bins
            def multi_bin_case():
                # Calculate overlap factors
                start_factor = ((old_edges[start+1] - new_edges[j]) / 
                               (old_edges[start+1] - old_edges[start]))
                end_factor = ((new_edges[j+1] - old_edges[stop]) / 
                             (old_edges[stop+1] - old_edges[stop]))
                
                # Create modified widths for overlap calculation
                widths_slice = old_widths[start:stop+1]
                widths_modified = widths_slice.at[0].multiply(start_factor)
                if stop > start:
                    widths_modified = widths_modified.at[-1].multiply(end_factor)
                
                # Calculate weighted flux
                flux_slice = old_fluxes[..., start:stop+1]
                weighted_flux = widths_modified * flux_slice
                total_width = jnp.sum(widths_modified)
                new_flux = jnp.sum(weighted_flux, axis=-1) / total_width
                
                # Calculate weighted errors if provided
                if old_errs is not None:
                    err_slice = old_errs[..., start:stop+1]
                    weighted_err_sq = (widths_modified * err_slice) ** 2
                    new_err = jnp.sqrt(jnp.sum(weighted_err_sq, axis=-1)) / total_width
                else:
                    new_err = None
                
                return new_flux, new_err
            
            # Choose between single and multi-bin cases
            is_single_bin = (stop == start)
            flux, err = lax.cond(is_single_bin, single_bin_case, multi_bin_case)
            return flux, err
        
        # Choose between fill and normal processing
        new_flux, new_err = lax.cond(outside_range, fill_bin, process_normal_bin)
        
        # Update arrays
        new_fluxes = new_fluxes.at[..., j].set(new_flux)
        if new_errs is not None and new_err is not None:
            new_errs = new_errs.at[..., j].set(new_err)
        
        return new_fluxes, new_errs
    
    # Process all bins using lax.fori_loop
    final_arrays = lax.fori_loop(0, len(new_wavs), process_bin, (new_fluxes, new_errs))
    new_fluxes, new_errs = final_arrays
    
    # Warning for out-of-bounds values (only works in non-JIT context)
    if verbose:
        out_of_bounds = ((new_edges[0] < old_edges[0]) | 
                        (new_edges[-1] > old_edges[-1]))
        if out_of_bounds:
            warnings.warn(
                "Spectres: new_wavs contains values outside the range "
                "in spec_wavs, new_fluxes and new_errs will be filled "
                "with the value set in the 'fill' keyword argument.",
                category=RuntimeWarning,
            )
    
    # Return results
    if old_errs is not None:
        return new_fluxes, new_errs
    else:
        return new_fluxes