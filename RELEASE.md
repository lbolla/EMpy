Release checklist (uv + setuptools_scm)
=====================================

This project uses `uv` for dependency management and `setuptools_scm` for
versioning. Follow these steps to cut a release:

1. Update `CHANGES` with release notes.

2. Create a git tag matching the version you want to publish (example: `v2.2.2`):

```bash
git tag vX.Y.Z
git push --tags
```

3. Regenerate the lockfile and build artifacts locally (this ensures the
   lockfile in the repo matches the built packages):

```bash
uv lock
uv run python -m build
```

4. Optionally run tests in the uv-managed environment:

```bash
uv venv --python 3.12
uv sync
uv run pytest
```

5. Push a release branch / open PR with the `CHANGES` and `uv.lock` updates.

6. In CI the workflow `.github/workflows/release.yml` will build artifacts
   on tag push. Publishing to PyPI is intentionally commented out in the
   workflow; add `TWINE_USERNAME` and `TWINE_PASSWORD` secrets and enable the
   `twine upload` step when you're ready.

Notes
-----
- `setuptools_scm` derives the version from git tags; tagging is required.
- Keep `MANIFEST.in` for data files; `pyproject.toml` has `include-package-data = true`.
