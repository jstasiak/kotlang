.PHONY: lint
lint: flake8 mypy

.PHONY: flake8
flake8:
	flake8

.PHONY: cloc
cloc:
	cloc --vcs git .

.PHONY: mypy
mypy:
	python -m mypy kotlang

.PHONY: test
test:
	py.test -v


.PHONY: ci
ci: test lint
