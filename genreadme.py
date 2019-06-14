import io
import os

outs = []
startstr = ".. currentmodule::"
includestr = '.. include::'
docdir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'docs')


def parse_file(filename):
    out_data = []
    with io.open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
        skipping = False
        for line in data:
            if startstr in line:
                skipping = True
                continue
            if skipping and line.strip() == '':
                continue
            else:
                skipping = False
            lstripline = line.lstrip()
            if lstripline.startswith(includestr):
                subfilename = lstripline[len(includestr):].strip()
                # [1:] to remove the leading / at the start; ends up being parsed as root
                out_line = parse_file(os.path.join(docdir, subfilename[1:]))
            else:
                out_line = line
            out_data.append(out_line)
    return ''.join(out_data)


def read_from_files(filenames):
    for filename in filenames:
        outs.append(parse_file(filename))


read_from_files(['./docs/fragments/title.rst', './docs/pages/info.rst', './docs/pages/installation.rst'])

outs.append("Documentation\n"
            "-------------\n"
            "The documentation is available `here <https://signatory.readthedocs.io>`_.")

read_from_files(['./docs/pages/faq.rst', './docs/pages/citation.rst', './docs/pages/acknowledgements.rst'])

with io.open('./README.rst', 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(outs))
