from synthesizer.parametric.galaxy import Galaxy as ParametricGalaxy
from synthesizer.fitter.priors import Common as CommonPrior

class Galaxy(ParametricGalaxy):
    """A class defining parametric galaxy objects"""

    def __init__(
        self,
        stars=None,
        black_holes=None,
        redshift=None,
        **kwargs,
    ):
        if isinstance(redshift, CommonPrior):
            redshift_prior = redshift
            redshift = redshift_prior.sample()
            # add to free_params dict? 

        super().__init__(
            stars=stars,
            name="fitted galaxy",
            black_holes=black_holes,
            redshift=redshift,
            centre=None,
            **kwargs,
        )