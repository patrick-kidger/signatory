import io

outs = []

def read_from_files(files):
    for file in files:
        with io.open(file, 'r', encoding='utf-8') as f:
            outs.append(f.read())

read_from_files(['./docs/fragments/title.rst', './docs/fragments/info.rst', './docs/fragments/installation.rst'])

outs.append("Documentation\n"
            "-------------\n"
            "The documentation is available `here <https://signatory.readthedocs.io>`_.")

read_from_files(['./docs/fragments/faq.rst', './docs/fragments/citation.rst', './docs/fragments/acknowledgements.rst'])

with io.open('./README.rst', 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(outs))
