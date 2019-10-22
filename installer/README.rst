*******************
Signatory Installer
*******************

This is the install script for the `Signatory <https://github.com/patrick-kidger/signatory>`__ project.

If you're trying to install Signatory then you probably want to do it via this script:

.. code-block::

    pip install signatory_installer

This wrapper script is to solve a particular issue: if you are using a precompiled version of Signatory, then it must have been compiled against the same version of PyTorch as you are using. And there's no progammatic way (that I know of!) to get the version of PyTorch currently installed, and download the corresponding appropriate version of Signatory.