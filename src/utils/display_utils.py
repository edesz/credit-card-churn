#!/usr/bin/env python3


"""Define helper functions to display a Python module in a notebook."""

import textwrap

from IPython.display import HTML, Markdown, display
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


def markdown_highlight_filtered(
    fpath: str, unwanted_ranges: list[tuple[int, int] | int]
) -> None:
    """Displays filtered, dedented Python source code in a notebook.

    This function reads a Python source file, excludes specified line
    ranges or individual lines, removes common leading indentation,
    and renders the result as a single Markdown code block within a
    Jupyter notebook.

    Args:
        fpath: The path (str) to the Python source file.
        unwanted_ranges: A list containing zero-based (start, end) line
            index tuples, individual integer line indices, or tuple
            concatenations to exclude from the display.

    Returns:
        None
    """
    unwanted_indices = set()
    for item in unwanted_ranges:
        if isinstance(item, int):
            unwanted_indices.add(item)
        elif isinstance(item, tuple):
            if len(item) == 1:
                unwanted_indices.add(item[0])  # Fix: Extract item[0]
            elif len(item) == 2:
                unwanted_indices.update(
                    range(item[0], item[1])
                )  # Fix: Use tuple items

    with open(fpath, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    kept_lines = [
        line
        for idx, line in enumerate(all_lines)
        if idx not in unwanted_indices
    ]
    code_str = "".join(kept_lines)
    dedented_code = textwrap.dedent(code_str).strip("\n")

    final_markdown_output = f"```python\n{dedented_code}\n```"
    display(Markdown(final_markdown_output))
