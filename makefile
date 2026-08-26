.PHONY: clean test pyright

all: test

test: .venv
	uv run pytest -s -v -x

.venv:
	uv sync

pyright: .venv
	uv run pyright valentbind

clean:
	rm -rf coverage.xml
