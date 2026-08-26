.PHONY: clean test ty

all: test

test: .venv
	uv run pytest -s -v -x

.venv:
	uv sync

ty: .venv
	uv run ty check valentbind

clean:
	rm -rf coverage.xml
