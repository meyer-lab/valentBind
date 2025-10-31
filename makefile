.PHONY: clean test pyright

all: test

test: .venv
	uv run pytest -s -v -x

.venv:
	uv sync

coverage.xml: .venv
	uv run pytest --junitxml=junit.xml --cov=valentbind --cov-report xml:coverage.xml

pyright: .venv
	uv run pyright valentbind

clean:
	rm -rf coverage.xml
