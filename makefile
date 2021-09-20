.PHONY: clean test

all: test

venv: venv/bin/activate

venv/bin/activate:
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install --user poetry
	touch venv/bin/activate

test: venv
	. venv/bin/activate && pytest -s -v -x

coverage.xml: venv
	. venv/bin/activate && pytest --junitxml=junit.xml --cov=valentbind --cov-report xml:coverage.xml

clean:
	rm -rf venv coverage.xml