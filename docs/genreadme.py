# GitHub doesn't support .. include::, so here we are.

import io


outs = []
with io.open('./title.rst', 'r', encoding='utf-8') as f:
    outs.append(f.read())

with io.open('./installation.rst', 'r', encoding='utf-8') as f:
    outs.append(f.read())

outs.append("Documentation\n"
            "-------------\n"
            "The documentation is available `here <https://signatory.readthedocs.io>`_.")

with io.open('./faq.rst', 'r', encoding='utf-8') as f:
    outs.append(f.read())

with io.open('./citation.rst', 'r', encoding='utf-8') as f:
    outs.append(f.read())

with io.open('./acknowledgements.rst', 'r', encoding='utf-8') as f:
    outs.append(f.read())

with io.open('../README.rst', 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(outs))
