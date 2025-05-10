
from difflib import SequenceMatcher
def check_spelling_against_list(string1, list_of_strings):
    """
    Checks the similarity between two strings using SequenceMatcher.
    """
    for string2 in list_of_strings:
        matcher = SequenceMatcher(None, string1, string2)
        if matcher.ratio() > 0.5:
            return string2

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

    bins = np.zeros(midpoints.shape[0]+1)
    if fix_low is not None:
        bins[0] = fix_low
    else:
        bins[0] = midpoints[0] - (midpoints[1]-midpoints[0])/2
    if fix_high is not None:
        bins[-1] = fix_high
    else:
        bins[-1] = midpoints[-1] + (midpoints[-1]-midpoints[-2])/2
    bins[1:-1] = (midpoints[1:] + midpoints[:-1])/2

    return bins
