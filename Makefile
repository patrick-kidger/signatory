.PHONY: genreadme setup test docs publish
.RECIPEPREFIX+=

# generate the readme from the documentation
genreadme:
    python genreadme.py

# run setup.py
setup:
    python setup.py ${O}

# run tests
test:
    python -m test.runner

# make docs
docs:
    sphinx-build -M html ./docs ./docs/_build

# push to pypi
publish:
    twine upload dist/*