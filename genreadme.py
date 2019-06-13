import io

outs = []
with io.open('./docs/fragments/title.rst', 'r', encoding='utf-8') as f:
    outs.append(f.read())

with io.open('./docs/fragments/info.rst', 'r', encoding='utf-8') as f:
    outs.append(f.read())

with io.open('./docs/pages/installation.rst', 'r', encoding='utf-8') as f:
    outs.append(f.read())

outs.append("Documentation\n"
            "-------------\n"
            "The documentation is available `here <https://signatory.readthedocs.io>`_.")

with io.open('./docs/pages/faq.rst', 'r', encoding='utf-8') as f:
    outs.append(f.read())

with io.open('./docs/pages/citation.rst', 'r', encoding='utf-8') as f:
    outs.append(f.read())

with io.open('./docs/pages/acknowledgements.rst', 'r', encoding='utf-8') as f:
    outs.append(f.read())

with io.open('./README.rst', 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(outs))
