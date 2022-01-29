.EXPORT_ALL_VARIABLES:
.PHONY: help

SRC = EMpy
SRC_TEST = tests
REQUIREMENTS = requirements.txt requirements_dev.txt

# Self-documenting Makefile
# https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help:  ## Print this help
	@grep -E '^[a-zA-Z][a-zA-Z0-9_-]*:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

develop: upgrade-dev requirements-install  ## Install project for development
	pip install -e .

upgrade-dev:  ## Upgrade packages for development
	pip install -U setuptools pip pip-tools tox

test: lint  ## Run tests
	python setup.py test

tox:  ## Run Python tests
	tox

black:  ## Run formatter
	black .

lint: flake8 pyflakes mypy  ## Run linters

flake8:  ## Run flake8 linter
	flake8 ${SRC} tests examples scripts

pyflakes:  ## Run pyflake linter
	pyflakes ${SRC} tests examples scripts

mypy:  ## Run mypy linter
	mypy ${SRC} tests examples scripts

requirements: ${REQUIREMENTS}  ## Create requirements files

requirements.txt: setup.py
	pip-compile -v ${PIP_COMPILE_ARGS} --output-file requirements.txt setup.py

%.txt: %.in
	pip-compile -v ${PIP_COMPILE_ARGS} --output-file $@ $<

requirements-upgrade: PIP_COMPILE_ARGS += --upgrade
requirements-upgrade: requirements  ## Upgrade requirements

requirements-sync: requirements  ## Synchronize requirements
	pip-sync ${REQUIREMENTS}
	pip install -e .

requirements-install: requirements  ## Install requirements
	$(foreach req, ${REQUIREMENTS}, pip install --no-binary :all: -r $(req);)

clean-repo:
	git diff --quiet HEAD  # no pending commits
	git diff --cached --quiet HEAD  # no unstaged changes
	git pull --ff-only  # latest code

release: requirements clean-repo  ## Make a release (specify: PART=[major|minor|patch])
	bump2version ${PART}
	git push
	git push --tags
