__all__ = [
    "check_spelling_against_list",
]
import jax.numpy as jnp

from difflib import SequenceMatcher
def check_spelling_against_list(string1, list_of_strings):
    """
    Checks the similarity between two strings using SequenceMatcher.
    """
    for string2 in list_of_strings:
        matcher = SequenceMatcher(None, string1, string2)
        if matcher.ratio() > 0.5:
            return string2
