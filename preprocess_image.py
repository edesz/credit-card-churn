#!/usr/bin/env python3


"""Trim whitespace and make background transparent for docs logo image."""

from collections import deque
from pathlib import Path

import typer
from PIL import Image
from typing_extensions import Annotated

app = typer.Typer(
    name="image-tools",
    help="CLI tools for cropping padding and generating transparent backgrounds.",
    add_completion=False,
)


def _is_background_pixel(r: int, g: int, b: int, tolerance: int) -> bool:
    """Helper to detect light/dark grey checkered background pixels."""
    return r > tolerance and g > tolerance and b > tolerance


def _trim_whitespace_img(img: Image.Image, tolerance: int) -> Image.Image:
    """Internal function to calculate bounding box and crop padding."""
    pixels = img.load()
    width, height = img.size

    min_x, min_y = width, height
    max_x, max_y = 0, 0

    for x in range(width):
        for y in range(height):
            r, g, b, _ = pixels[x, y]
            if not _is_background_pixel(r, g, b, tolerance):
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y

    if min_x < max_x and min_y < max_y:
        return img.crop((min_x, min_y, max_x + 1, max_y + 1))
    return img


def _make_transparent_img(img: Image.Image, tolerance: int) -> Image.Image:
    """Internal function to flood-fill background starting from outer corners."""
    pixels = img.load()
    width, height = img.size

    queue = deque(
        [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]
    )
    visited = set(queue)

    while queue:
        x, y = queue.popleft()
        r, g, b, _ = pixels[x, y]

        if _is_background_pixel(r, g, b, tolerance):
            pixels[x, y] = (255, 255, 255, 0)

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < width
                    and 0 <= ny < height
                    and (nx, ny) not in visited
                ):
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    return img


@app.command()
def trim_whitespace(
    input_path: Annotated[
        Path,
        typer.Argument(
            exists=True, file_okay=True, dir_okay=False, readable=True
        ),
    ],
    output_path: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "trimmed_image.png"
    ),
    tolerance: Annotated[
        int, typer.Option("--tolerance", "-t", min=0, max=255)
    ] = 130,
) -> None:
    """Crop exterior whitespace/padding surrounding the central object."""
    try:
        img = Image.open(input_path).convert("RGBA")
        cropped_img = _trim_whitespace_img(img, tolerance)
        cropped_img.save(output_path, "PNG")
        typer.secho(
            f"Successfully trimmed whitespace: {output_path}",
            fg=typer.colors.GREEN,
        )
    except Exception as e:
        typer.secho(
            f"Error trimming whitespace: {e}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)


@app.command()
def make_transparent(
    input_path: Annotated[
        Path,
        typer.Argument(
            exists=True, file_okay=True, dir_okay=False, readable=True
        ),
    ],
    output_path: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "transparent_image.png"
    ),
    tolerance: Annotated[
        int, typer.Option("--tolerance", "-t", min=0, max=255)
    ] = 130,
) -> None:
    """Flood-fill exterior background pixels to make them transparent."""
    try:
        img = Image.open(input_path).convert("RGBA")
        transparent_img = _make_transparent_img(img, tolerance)
        transparent_img.save(output_path, "PNG")
        typer.secho(
            f"Successfully made background transparent: {output_path}",
            fg=typer.colors.GREEN,
        )
    except Exception as e:
        typer.secho(
            f"Error making background transparent: {e}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@app.command()
def preprocess(
    input_path: Annotated[
        Path,
        typer.Argument(
            exists=True, file_okay=True, dir_okay=False, readable=True
        ),
    ],
    output_path: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "hexagon_logo_final.png"
    ),
    tolerance: Annotated[
        int, typer.Option("--tolerance", "-t", min=0, max=255)
    ] = 130,
) -> None:
    """Pipeline command: First trims whitespace around image, then makes background transparent."""
    try:
        typer.echo(f"Processing pipeline for: {input_path}")
        img = Image.open(input_path).convert("RGBA")

        # Step 1: Trim Whitespace
        typer.echo("Step 1/2: Trimming whitespace...")
        img = _trim_whitespace_img(img, tolerance)

        # Step 2: Make Background Transparent
        typer.echo("Step 2/2: Converting background to transparent...")
        img = _make_transparent_img(img, tolerance)

        img.save(output_path, "PNG")
        typer.secho(
            f"Successfully processed final image: {output_path}",
            fg=typer.colors.GREEN,
        )

    except Exception as e:
        typer.secho(f"Pipeline failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
