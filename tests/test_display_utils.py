#!/usr/bin/env python3


"""Test pygments utilities."""

import builtins
from unittest.mock import mock_open, patch

import src.utils.display_utils as du
from src.utils.display_utils import markdown_highlight_filtered


def test_pygments_highlight_basic(monkeypatch, sample_code):
    """Verifies function reads file and calls display."""
    mock_file = mock_open(read_data=sample_code)
    monkeypatch.setattr(builtins, "open", mock_file)

    captured = {}

    def mock_display(obj):
        captured["called"] = True
        captured["html"] = obj.data

    monkeypatch.setattr(du, "display", mock_display)

    du.pygments_highlight("fake.py", unwanted_lines=[])

    assert captured.get("called", False) is True
    assert "<style>" in captured["html"]
    assert "print" in captured["html"]


def test_unwanted_lines_removed(monkeypatch, sample_code):
    """Verifies unwanted lines are removed from output."""
    mock_file = mock_open(read_data=sample_code)
    monkeypatch.setattr(builtins, "open", mock_file)

    captured = {}

    def mock_display(obj):
        captured["html"] = obj.data

    monkeypatch.setattr(du, "display", mock_display)

    # Remove second line (index 1)
    du.pygments_highlight("fake.py", unwanted_lines=[1])

    html = captured["html"]

    assert "hello" in html
    assert "!" in html
    assert "world" not in html


def test_style_is_applied(monkeypatch, sample_code):
    """Verifies custom Pygments style is applied.

    Args:
        monkeypatch: Pytest fixture.
        sample_code: Sample file content.

    Returns:
        None
    """
    import builtins
    from unittest.mock import mock_open

    mock_file = mock_open(read_data=sample_code)
    monkeypatch.setattr(builtins, "open", mock_file)

    captured = {}

    def mock_display(obj):
        captured["html"] = obj.data

    monkeypatch.setattr(du, "display", mock_display)

    du.pygments_highlight("fake.py", unwanted_lines=[], style="monokai")

    html = captured["html"]

    # test that style block is present and highlighting applied
    assert "<style>" in html
    assert "highlight" in html


def test_empty_file(monkeypatch):
    """Verifies function handles empty file gracefully.

    Args:
        monkeypatch: Pytest fixture.

    Returns:
        None
    """
    mock_file = mock_open(read_data="")
    monkeypatch.setattr(builtins, "open", mock_file)

    captured = {}

    def mock_display(obj):
        captured["called"] = True

    monkeypatch.setattr(du, "display", mock_display)

    du.pygments_highlight("fake.py", unwanted_lines=[])

    assert captured.get("called", False) is True


def test_example_usage(monkeypatch, sample_code):
    """Example usage of pygments_highlight in practice.

    This test demonstrates how a user would call the function
    to render syntax-highlighted code while excluding lines.

    Args:
        monkeypatch: Pytest fixture.
        sample_code: Sample file content.

    Returns:
        None
    """
    mock_file = mock_open(read_data=sample_code)
    monkeypatch.setattr(builtins, "open", mock_file)

    calls = {"count": 0}

    def mock_display(_):
        calls["count"] += 1

    monkeypatch.setattr(du, "display", mock_display)

    # test removal of middle line
    du.pygments_highlight(
        fpath="example.py",
        unwanted_lines=[1],
        style="default",
    )

    assert calls["count"] == 1


def test_excludes_individual_line():
    """Excludes a single line specified by its zero-based index."""
    source = (
        "def hello():\n"
        "    print('hello')\n"
        "    print('unwanted')\n"
        "    return True\n"
    )

    expected = (
        "```python\n"
        "def hello():\n"
        "    print('hello')\n"
        "    return True\n"
        "```"
    )

    with (
        patch(
            "src.utils.display_utils.open",
            mock_open(read_data=source),
        ),
        patch("src.utils.display_utils.display") as mock_display,
    ):
        markdown_highlight_filtered("example.py", [2])

    mock_display.assert_called_once()
    markdown_object = mock_display.call_args.args[0]

    assert markdown_object.data == expected


def test_excludes_line_range():
    """Excludes all lines within a specified zero-based range."""
    source = (
        "def hello():\n"
        "    print('one')\n"
        "    print('two')\n"
        "    print('three')\n"
        "    return True\n"
    )

    expected = "```python\n" "def hello():\n" "    return True\n" "```"

    with (
        patch(
            "src.utils.display_utils.open",
            mock_open(read_data=source),
        ),
        patch("src.utils.display_utils.display") as mock_display,
    ):
        markdown_highlight_filtered("example.py", [(1, 4)])

    markdown_object = mock_display.call_args.args[0]

    assert markdown_object.data == expected


def test_excludes_single_element_tuple():
    """Excludes a line specified by a single-element tuple."""
    source = "def hello():\n" "    print('hello')\n" "    print('unwanted')\n"

    expected = "```python\n" "def hello():\n" "    print('hello')\n" "```"

    with (
        patch(
            "src.utils.display_utils.open",
            mock_open(read_data=source),
        ),
        patch("src.utils.display_utils.display") as mock_display,
    ):
        markdown_highlight_filtered("example.py", [(2,)])

    markdown_object = mock_display.call_args.args[0]

    assert markdown_object.data == expected


def test_combines_integer_and_range_exclusions():
    """Combines individual line and range exclusions."""
    source = "line 0\n" "line 1\n" "line 2\n" "line 3\n" "line 4\n" "line 5\n"

    expected = "```python\n" "line 0\n" "line 3\n" "line 5\n" "```"

    with (
        patch(
            "src.utils.display_utils.open",
            mock_open(read_data=source),
        ),
        patch("src.utils.display_utils.display") as mock_display,
    ):
        markdown_highlight_filtered("example.py", [1, (2, 3), (4, 5)])

    markdown_object = mock_display.call_args.args[0]

    assert markdown_object.data == expected


def test_dedents_common_leading_indentation():
    """Removes common leading indentation from the source code."""
    source = (
        "    def hello():\n" "        print('hello')\n" "        return True\n"
    )

    expected = (
        "```python\n"
        "def hello():\n"
        "    print('hello')\n"
        "    return True\n"
        "```"
    )

    with (
        patch(
            "src.utils.display_utils.open",
            mock_open(read_data=source),
        ),
        patch("src.utils.display_utils.display") as mock_display,
    ):
        markdown_highlight_filtered("example.py", [])

    markdown_object = mock_display.call_args.args[0]

    assert markdown_object.data == expected


def test_strips_leading_and_trailing_newlines():
    """Removes leading and trailing newlines from the rendered code."""
    source = "\n\n    print('hello')\n\n"

    expected = "```python\n" "print('hello')\n" "```"

    with (
        patch(
            "src.utils.display_utils.open",
            mock_open(read_data=source),
        ),
        patch("src.utils.display_utils.display") as mock_display,
    ):
        markdown_highlight_filtered("example.py", [])

    markdown_object = mock_display.call_args.args[0]

    assert markdown_object.data == expected


def test_empty_source():
    """Renders an empty source file as an empty Python code block."""
    with (
        patch(
            "src.utils.display_utils.open",
            mock_open(read_data=""),
        ),
        patch("src.utils.display_utils.display") as mock_display,
    ):
        markdown_highlight_filtered("example.py", [])

    markdown_object = mock_display.call_args.args[0]

    assert markdown_object.data == "```python\n\n```"


def test_opens_file_as_utf8():
    """Opens the source file in read mode using UTF-8 encoding."""
    source = "print('hello')\n"

    with (
        patch(
            "src.utils.display_utils.open",
            mock_open(read_data=source),
        ) as mock_file,
        patch("src.utils.display_utils.display"),
    ):
        markdown_highlight_filtered("example.py", [])

    mock_file.assert_called_once_with(
        "example.py",
        "r",
        encoding="utf-8",
    )


def test_ignores_empty_tuple():
    """Ignores an empty tuple in the unwanted ranges."""
    source = "line 0\n" "line 1\n" "line 2\n"

    expected = "```python\n" "line 0\n" "line 1\n" "line 2\n" "```"

    with (
        patch(
            "src.utils.display_utils.open",
            mock_open(read_data=source),
        ),
        patch("src.utils.display_utils.display") as mock_display,
    ):
        markdown_highlight_filtered("example.py", [()])

    markdown_object = mock_display.call_args.args[0]

    assert markdown_object.data == expected


def test_ignores_tuple_with_more_than_two_elements():
    """Ignores tuples containing more than two elements."""
    source = "line 0\n" "line 1\n" "line 2\n"

    expected = "```python\n" "line 0\n" "line 1\n" "line 2\n" "```"

    with (
        patch(
            "src.utils.display_utils.open",
            mock_open(read_data=source),
        ),
        patch("src.utils.display_utils.display") as mock_display,
    ):
        markdown_highlight_filtered("example.py", [(0, 1, 2)])

    markdown_object = mock_display.call_args.args[0]

    assert markdown_object.data == expected


def test_ignores_non_integer_non_tuple_item():
    """Ignores unwanted range items that are not integers or tuples."""
    source = "line 0\n" "line 1\n" "line 2\n"

    expected = "```python\n" "line 0\n" "line 1\n" "line 2\n" "```"

    with (
        patch(
            "src.utils.display_utils.open",
            mock_open(read_data=source),
        ),
        patch("src.utils.display_utils.display") as mock_display,
    ):
        markdown_highlight_filtered("example.py", ["invalid"])

    markdown_object = mock_display.call_args.args[0]

    assert markdown_object.data == expected
