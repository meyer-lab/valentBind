.PHONY: clean test pyright

all: test

test: .venv
	rye run pytest -s -v -x

.venv:
	rye sync

coverage.xml: .venv
	rye run pytest --junitxml=junit.xml --cov=valentbind --cov-report xml:coverage.xml

pyright: .venv
	rye run pyright valentbind

clean:
	rm -rf coverage.xml