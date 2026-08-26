.PHONY: clean test ty

all: test

test: .venv
	uv run pytest -s -v -x

.venv:
	uv sync

coverage.xml: .venv
	uv run pytest --junitxml=junit.xml --cov=valentbind --cov-report xml:coverage.xml

ty: .venv
	uv run ty check valentbind

clean:
	rm -rf coverage.xml
