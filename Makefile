.PHONY: help serve build clean

PORT ?= 4000
BASEURL ?= /EMpy

# Self-documenting Makefile
help:  ## Print this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

serve:  ## Serve the Jekyll website locally via Docker at http://localhost:4000/EMpy/
	docker run --rm -it \
		-p $(PORT):4000 \
		-v "$(CURDIR):/srv/jekyll" \
		-v empy-jekyll-bundle:/usr/local/bundle \
		jekyll/builder:4 \
		bash -lc 'printf "source \"https://rubygems.org\"\ngem \"github-pages\", group: :jekyll_plugins\ngem \"webrick\"\n" > /usr/local/bundle/Gemfile && (BUNDLE_GEMFILE=/usr/local/bundle/Gemfile bundle check || BUNDLE_GEMFILE=/usr/local/bundle/Gemfile bundle install) && BUNDLE_GEMFILE=/usr/local/bundle/Gemfile bundle exec jekyll serve --host 0.0.0.0 --baseurl $(BASEURL)'

build:  ## Build the static site into _site/ via Docker
	docker run --rm \
		-v "$(CURDIR):/srv/jekyll" \
		-v empy-jekyll-bundle:/usr/local/bundle \
		jekyll/builder:4 \
		bash -lc 'printf "source \"https://rubygems.org\"\ngem \"github-pages\", group: :jekyll_plugins\n" > /usr/local/bundle/Gemfile && (BUNDLE_GEMFILE=/usr/local/bundle/Gemfile bundle check || BUNDLE_GEMFILE=/usr/local/bundle/Gemfile bundle install) && BUNDLE_GEMFILE=/usr/local/bundle/Gemfile bundle exec jekyll build --baseurl $(BASEURL)'

clean:  ## Remove generated _site/ and Jekyll caches
	docker run --rm -v "$(CURDIR):/srv/jekyll" jekyll/builder:4 rm -rf /srv/jekyll/_site /srv/jekyll/.jekyll-cache /srv/jekyll/.jekyll-metadata /srv/jekyll/.sass-cache
