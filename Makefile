.EXPORT_ALL_VARIABLES:
.PHONY: help

SRC = EMpy
REQUIREMENTS = requirements.txt requirements_dev.txt

# Self-documenting Makefile
# https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help:  ## Print this help
	@grep -E '^[a-zA-Z][a-zA-Z0-9_-]*:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

develop: upgrade-setuptools upgrade-pip upgrade-pip-tools requirements-install  ## Install project for development
	pip install -e .

upgrade-setuptools:  ## Upgrade setuptools
	pip install -U setuptools

upgrade-pip:  ## Upgrade pip
	pip install -U pip

upgrade-pip-tools:  ## Upgrade pip-tools
	pip install -U pip-tools

test: tox lint  ## Run tests

tox:  ## Run Python tests
	tox

lint: flake8 mypy  ## Run linters

flake8:  ## Run flake8 linter
	flake8 ${SRC}

mypy:  ## Run mypy linter
	mypy ${SRC}

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
	$(foreach req, ${REQUIREMENTS}, pip install -r $(req);)

clean-repo:
	git diff --quiet HEAD  # no pending commits
	git diff --cached --quiet HEAD  # no unstaged changes
	git pull --ff-only  # latest code

release: requirements clean-repo  ## Make a release (specify: PART=[major|minor|patch])
	bump2version ${PART}
	git push
	git push --tags
