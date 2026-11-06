# Documentation

This site is built with [Zensical](https://zensical.org/) from the Markdown files in `docs/` and the configuration in
`zensical.toml`. The [API reference](../reference/index.md) is generated from the docstrings by
[mkdocstrings](https://mkdocstrings.github.io/), which reads the source code without importing it.

## Preview and build

The documentation tools are in the `docs` dependency group and need Python 3.11 or newer. They do not need YAIB's
runtime dependencies, so the commands below install only that group, into a separate environment:

```bash
UV_PROJECT_ENVIRONMENT=.venv-docs uv run --only-group docs zensical serve
```

This serves the site at <http://localhost:8000> and rebuilds it when a page changes. To build the static site into
`site/`:

```bash
UV_PROJECT_ENVIRONMENT=.venv-docs uv run --only-group docs zensical build --clean
```

The separate environment keeps the docs-only sync from removing packages from the main `.venv`.

## Publishing

The `Documentation` workflow (`.github/workflows/docs.yml`) builds the site for every pull request and publishes it to
GitHub Pages on every push to `development`.

## Writing pages

- Add new pages to `nav` in `zensical.toml`.
- Show config files and code from the repository with snippets instead of copying them, so the page cannot go stale.
    Inside a code block, use the block form, with the path on its own line between two `--8<--` lines: the one-line form
    `--8<-- "path"` parses as Python, and the mdformat pre-commit hook would reformat it. Paths are relative to the
    repository root, and a missing file fails the build.
- Use admonitions (`!!! note`, `!!! warning`) for side remarks and content tabs (`=== "R"`) for alternatives.
- Link to other pages with relative paths to their `.md` files.

## Docstrings

Write docstrings in the Google style, so mkdocstrings can render the arguments and return values:

```python
def preprocess_data(data_dir: Path, seed: int = 42) -> dict:
    """Load the cohort and split and preprocess it for one fold.

    Args:
        data_dir: Folder with the cohort's parquet files.
        seed: Random seed for the splits.

    Returns:
        The preprocessed data per split and segment.
    """
```

To document a new module, add a `::: icu_benchmarks.<module>` line to the matching page in `docs/reference/`.
