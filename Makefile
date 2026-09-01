.PHONY: install test smoke preflight download-data prepare-data report

install:
	python3 -m pip install -e .

test:
	python3 -m pytest -q

smoke:
	python3 -m fedqtrust smoke-test --download-data --device auto --output-dir output/smoke_test

preflight:
	python3 -m fedqtrust preflight --profile paper --device auto --output-dir output

download-data:
	python3 -m fedqtrust download-data --output-dir output

prepare-data:
	python3 -m fedqtrust prepare-data --output-dir output

report:
	python3 -m fedqtrust generate-report --output-dir output

