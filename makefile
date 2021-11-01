.PHONY: clean test

all: test

test:
	poetry run pytest -s -v -x

coverage.xml:
	poetry run pytest --junitxml=junit.xml --cov=valentbind --cov-report xml:coverage.xml

mypy:
	poetry run mypy --install-types --non-interactive --ignore-missing-imports valentbind

clean:
	rm -rf coverage.xml