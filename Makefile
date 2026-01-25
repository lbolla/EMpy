.EXPORT_ALL_VARIABLES:
.PHONY: help

UV = uv

SRC = EMpy tests examples scripts

# Self-documenting Makefile
# https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help:  ## Print this help
	@grep -E '^[a-zA-Z][a-zA-Z0-9_-]*:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

develop:  ## Install all dependencies
	$(UV) sync

upgrade:  ## Upgrade dependencies
	$(UV) lock --upgrade
	$(UV) sync

test:  ## Run tests
	$(UV) run pytest

black:  ## Run formatter
	$(UV) run black ${SRC}

lint: flake8 pyflakes mypy  ## Run linters

flake8:  ## Run flake8 linter
	$(UV) run flake8 ${SRC}

pyflakes:  ## Run pyflake linter
	$(UV) run pyflakes ${SRC}

mypy:  ## Run mypy linter
	$(UV) run mypy ${SRC}

clean-repo:
	git diff --quiet HEAD  # no pending commits
	git diff --cached --quiet HEAD  # no unstaged changes
	git pull --ff-only  # latest code

release: clean-repo  ## Make a release (specify: PART=[major|minor|patch])
	bump2version ${PART}
	git push
	git push --tags
