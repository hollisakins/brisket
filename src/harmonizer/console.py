'''
This module provides utilities for logging and console output, using the ``rich`` and ``logging`` packages. 
'''

from rich.console import Console
console = Console()
console._log_render.omit_repeated_times = False

import logging
from rich.logging import RichHandler
def setup_logger(name, verbose):
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(fmt="%(message)s", datefmt="[%X]")
    handler = RichHandler(omit_repeated_times=False, rich_tracebacks=True, markup=True)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger

def rich_str(x):
    with console.capture() as capture:
        console.print(x, end="")
    return capture.get()

import re
from rich.highlighter import Highlighter
class PathHighlighter(Highlighter):
    def highlight(self, text):
        textstr = str(text)
        splitstr = re.split("(/)", textstr)
        if len(splitstr)==1:
            styledef = ['#FFE4B5']
        elif len(splitstr)==3:
            styledef = ['#6495ED','white','bold #FFE4B5']
        else:
            styledef = ['#6495ED','white','#8FBC8F','white','bold #FFE4B5']
        ltot = 0
        for i in range(len(splitstr)):
            text.stylize(styledef[i], ltot, ltot+len(splitstr[i]))
            ltot += +len(splitstr[i])

class LimitsHighlighter(Highlighter):
    def highlight(self, text):
        textstr = str(text)
        assert textstr[0] == '(' and textstr[-1] == ')'
        splitstr = ['(', textstr[1:-1].split(',')[0], ',', textstr[1:-1].split(',')[1], ')']
        ltot = 0
        styledef = ['white', 'bold #FFE4B5', 'white', 'bold #FFE4B5', 'white']
        for i in range(len(splitstr)):
            text.stylize(styledef[i], ltot, ltot+len(splitstr[i]))
            ltot += +len(splitstr[i])

        # text.stylize("bold red", 0, 4)

        # for index in range(len(text)):
        #     if str(text[index]) == "/":
        #         # yield index, index + 1, "bold red"
        #         text.stylize(f"bold red", index, index + 1)

# l = LimitsHighlighter()
# console.log(l('(9,11)'))