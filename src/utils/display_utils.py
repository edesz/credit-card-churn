#!/usr/bin/env python3


"""Define helper functions to display a Python module in a notebook."""

from IPython.display import HTML, display
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer


def pygments_highlight(
    fpath: str, unwanted_lines: list[int], style: str = "default"
) -> None:
    """Displays syntax-highlighted Python source code in a notebook.

    This function reads a Python source file, removes specified lines,
    applies syntax highlighting using Pygments, and renders the result
    as styled HTML within a Jupyter notebook.

    It is useful for presenting cleaned or simplified code examples in
    notebooks without modifying the original source file.

    Args:
        fpath: The path (str) to the Python source file.
        unwanted_lines: The zero-based line indices to exclude from display.
        style: The pygments highlighting style name.

    Returns:
        None

    Raises:
        FileNotFoundError: If the file path does not exist.
        IOError: If the file cannot be read.

    Examples:
        >>> pygments_highlight(
        ...     "src/module.py", unwanted_lines=[0, 1], style="friendly"
        ... )
    """
    with open(fpath, "r") as f:
        lines = f.readlines()

    filtered_lines = [
        line for i, line in enumerate(lines) if i not in unwanted_lines
    ]
    code = "".join(filtered_lines)

    formatter = HtmlFormatter(style=style, full=True)
    html_code = highlight(code, PythonLexer(), formatter)
    display(
        HTML(
            f"<style>{formatter.get_style_defs('.highlight')}</style>"
            f"{html_code}"
        )
    )
