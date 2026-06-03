# Contributing

When contributing to this repository, please first discuss the change you wish to make via an issue, email, or another method with the maintainers before making a change.

Please note that this project has a code of conduct. Please follow it in all interactions with the project.

## Development setup

YAIB uses `uv` together with `pyproject.toml` for environment and dependency management.

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create and sync the development environment:

```bash
uv python install 3.12
uv sync --dev
```

Run commands through the project environment with `uv run`, for example:

```bash
uv run python -m pytest
uv run ruff format --line-length 127
uv run ruff check --line-length 127 --statistics
```

## Autoformat and lint

For development purposes, YAIB uses `ruff` for formatting and linting:

```bash
uv run ruff format --line-length 127
uv run ruff check --line-length 127 --statistics
```

## Pull Request Process

1. Ensure documentation is updated for any interface changes, including installation, environment variables, paths, and configuration changes.
2. Update `README.md`, `PAPER.md`, or wiki pages when setup or usage instructions change.
3. Keep `pyproject.toml` and `uv.lock` in sync when dependencies change.
4. Use [SemVer](http://semver.org/) for versioning-related updates.
5. You may merge a pull request once you have the required reviewer approval, or request a reviewer to merge it for you if you do not have permission.

# YAIB release flow with uv build

## Trusted publishers to configure

Create two trusted publisher entries:

### On PyPI

- Owner: `rvandewater`
- Repository: `YAIB`
- Workflow: `python-build.yml`
- Environment: `pypi`

### On TestPyPI

- Owner: `rvandewater`
- Repository: `YAIB`
- Workflow: `python-build.yml`
- Environment: `testpypi`

The repository name and workflow filename must match exactly.

## What this workflow does

- Push to `development`:
    - builds with `uv build`
    - publishes to TestPyPI
- Push of a tag like `v1.0.2` or `1.0.2`:
    - builds with `uv build`
    - publishes to PyPI
    - creates a GitHub release and uploads signed artifacts

## How to release a development build

Push to the `development` branch:

```bash
git checkout development
git pull
git push origin development
```

That triggers a build and upload to TestPyPI.

## How to release a new production version

Because `uv build` uses the version already present in `pyproject.toml`, first update the package version there.

Example:

```toml
[project]
version = "1.0.2"
```

Then commit, tag, and push:

```bash
git checkout main
git pull

git add pyproject.toml uv.lock
git commit -m "Release 1.0.2"
git tag v1.0.2
git push origin main
git push origin v1.0.2
```

If you use plain numeric tags instead of `v`-prefixed tags, this also works:

```bash
git tag 1.0.2
git push origin 1.0.2
```

## Recommended release sequence

1. Merge the tested changes from `development` into `main`.
2. Update `version` in `pyproject.toml`.
3. Commit the release bump.
4. Create and push the tag.
5. Wait for the PyPI publish and GitHub release workflow to finish.

## Important note

With `uv build`, the tag itself does not set the package version. The version must already be defined in `pyproject.toml` before the build runs.

## Code of Conduct

### Our Pledge

In the interest of fostering an open and welcoming environment, contributors and maintainers pledge to make participation in this project and community a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

Examples of unacceptable behavior by participants include:

- The use of sexualized language or imagery and unwelcome sexual attention or advances
- Trolling, insulting or derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information, such as a physical or electronic address, without explicit permission
- Other conduct that could reasonably be considered inappropriate in a professional setting

### Our Responsibilities

Project maintainers are responsible for clarifying the standards of acceptable behavior and are expected to take appropriate and fair corrective action in response to any instances of unacceptable behavior.

Project maintainers have the right and responsibility to remove, edit, or reject comments, commits, code, wiki edits, issues, and other contributions that are not aligned with this Code of Conduct, or to ban temporarily or permanently any contributor for other behavior they deem inappropriate, threatening, offensive, or harmful.

### Scope

This Code of Conduct applies both within project spaces and in public spaces when an individual is representing the project or its community. Examples of representing a project or community include using an official project email address, posting via an official social media account, or acting as an appointed representative at an online or offline event. Representation of a project may be further defined and clarified by project maintainers.

### Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage], version 1.4, available at [http://contributor-covenant.org/version/1/4][version].

[homepage]: http://contributor-covenant.org
[version]: http://contributor-covenant.org/version/1/4/
