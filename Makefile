.PHONY: genreadme test docs publish prepublish buildandtest
.RECIPEPREFIX+=

# generate the readme from the documentation
genreadme:
    python genreadme.py

# run tests
test:
    python -m test.runner

# make docs
docs:
    sphinx-build -M html ./docs ./docs/_build


# push to pypi
publish:
    twine upload dist/*

# build and test everything
prepublish:
    ( \
    set -e ;\
    rm -rf dist ;\
    make genreadme ;\
    VERSION=$(python -c "import metadata; print(metadata.version)") ;\
    python setup.py sdist ;\
    make O=2.7 DIST=signatory-${VERSION}-cp27-cp27m-linux_x86_64.whl buildandtest ;\
    make O=3.5 DIST=signatory-${VERSION}-cp35-cp35m-linux_x86_64.whl buildandtest ;\
    make O=3.6 DIST=signatory-${VERSION}-cp36-cp36m-linux_x86_64.whl buildandtest ;\
    make O=3.7 DIST=signatory-${VERSION}-cp37-cp37m-linux_x86_64.whl buildandtest ;\
    make O=2.7 DIST=signatory-${VERSION}.tar.gz buildandtest ;\
    make O=3.5 DIST=signatory-${VERSION}.tar.gz buildandtest ;\
    make O=3.6 DIST=signatory-${VERSION}.tar.gz buildandtest ;\
    make O=3.7 DIST=signatory-${VERSION}.tar.gz buildandtest ;\
    )

#### Not to call directly ####

# build and test on a particular version
buildandtest:
    ( \
    set -e ;\
<<<<<<< HEAD
    conda create --prefix=/tmp/signatory-$${O} -y python=$${O} ;\
=======
    conda create --prefix=/tmp/signatory-$${O} -y python=$${O};\
>>>>>>> 25da8ddaf5170784653c8e00cb76028a5f0db16b
    conda activate /tmp/signatory-$${O} ;\
    conda install -y pytorch==1.0.1 -c pytorch ;\
    python setup.py bdist_wheel ;\
    pip install dist/$${DIST} ;\
    pip install iisignature ;\
    make test ;\
    conda deactivate ;\
    conda env remove /tmp/signatory-$${O} ;\
    )