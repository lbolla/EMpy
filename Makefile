.EXPORT_ALL_VARIABLES:
.PHONY: help requirements-dev-upgrade

UV = uv

SRC = EMpy
SRC_TEST = tests
REQUIREMENTS = requirements.txt requirements_dev.txt

# Self-documenting Makefile
# https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help:  ## Print this help
	@grep -E '^[a-zA-Z][a-zA-Z0-9_-]*:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

venv:  ## Create venv for EMpy (uv-managed)
	$(UV) venv

develop: upgrade-dev requirements-install  ## Install project for development
	$(UV) venv
	$(UV) sync
	$(UV) run pip install -e .

upgrade-dev:  ## Upgrade packages for development (run inside uv env)
	$(UV) run pip install -U setuptools pip pip-tools pytest tox wheel

test: lint  ## Run tests
	$(UV) run pytest

tox:  ## Run Python tests
	$(UV) run tox

black:  ## Run formatter
	$(UV) run black .

lint: flake8 pyflakes mypy  ## Run linters

flake8:  ## Run flake8 linter
	$(UV) run flake8 ${SRC} tests examples scripts

pyflakes:  ## Run pyflake linter
	$(UV) run pyflakes ${SRC} tests examples scripts

mypy:  ## Run mypy linter
	$(UV) run mypy ${SRC} tests examples scripts

requirements: ${REQUIREMENTS}  ## Create requirements files


requirements.txt: setup.py
	$(UV) pip compile ${PIP_COMPILE_ARGS} --output-file requirements.txt setup.py

%.txt: %.in
	$(UV) pip compile ${PIP_COMPILE_ARGS} --output-file $@ $<

requirements-upgrade: PIP_COMPILE_ARGS += --upgrade
requirements-upgrade: requirements  ## Upgrade requirements


requirements-sync: requirements  ## Synchronize requirements
	$(UV) sync
	$(UV) run pip install -e .


requirements-install: requirements  ## Install requirements
	$(foreach req, ${REQUIREMENTS}, $(UV) run pip install --no-binary :all: -r $(req);)

clean-repo:
	git diff --quiet HEAD  # no pending commits
	git diff --cached --quiet HEAD  # no unstaged changes
	git pull --ff-only  # latest code

release: requirements clean-repo  ## Make a release (specify: PART=[major|minor|patch])
	bump2version ${PART}
	git push
	git push --tags
