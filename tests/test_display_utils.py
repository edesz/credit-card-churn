#!/usr/bin/env python3


"""Test pygments utilities."""

import builtins
from unittest.mock import mock_open

import src.utils.display_utils as du


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
