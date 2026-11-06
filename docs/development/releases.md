# Releases

YAIB is built with `uv build` and published by the `python-build.yml` workflow.

| Event                                     | Result                                                                       |
| ----------------------------------------- | ---------------------------------------------------------------------------- |
| Push to `development`                     | Build and publish to TestPyPI                                                |
| Push of a tag such as `v1.0.2` or `1.0.2` | Build and publish to PyPI, and create a GitHub release with signed artifacts |

## Release a new version

`uv build` takes the version from `pyproject.toml`; the tag does not set it. So update the version first:

```toml
[project]
version = "1.0.2"
```

Then:

1. Merge the tested changes from `development` into `main`.

2. Update `version` in `pyproject.toml` and run `uv lock`.

3. Commit the version bump.

4. Create and push the tag:

    ```bash
    git checkout main
    git pull
    git add pyproject.toml uv.lock
    git commit -m "Release 1.0.2"
    git tag v1.0.2
    git push origin main
    git push origin v1.0.2
    ```

5. Wait for the PyPI publish and the GitHub release workflow to finish.

## Trusted publishers

Publishing uses PyPI's trusted publishers. Both entries must match the repository and workflow exactly:

| Index    | Owner         | Repository | Workflow           | Environment |
| -------- | ------------- | ---------- | ------------------ | ----------- |
| PyPI     | `rvandewater` | `YAIB`     | `python-build.yml` | `pypi`      |
| TestPyPI | `rvandewater` | `YAIB`     | `python-build.yml` | `testpypi`  |
