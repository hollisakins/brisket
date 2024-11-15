from rich.console import Console
console = Console()

def rich_str(x):
    with console.capture() as capture:
        console.print(x)
    return capture.get()