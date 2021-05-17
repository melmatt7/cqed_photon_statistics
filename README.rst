cqed_sims
================

.. image:: https://github.com/johnthagen/python-blueprint/workflows/python/badge.svg
    :target: https://github.com/melmatt7/cqed_photon_statistics/actions

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://black.readthedocs.io/en/stable/

.. image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
    :target: https://timothycrosley.github.io/isort/

Example Python project that demonstrates how to create a tested Python package using the latest
Python testing and linting tooling. The project contains a ``fact`` package that provides a
simple implementation of the `factorial algorithm <https://en.wikipedia.org/wiki/Factorial>`_
(``fact.lib``) and a command line interface (``fact.cli``).

Requirements
------------

Python 3.6+.

.. note::

    Because `Python 2.7 support ended January 1, 2020 <https://pythonclock.org/>`_, new projects
    should consider supporting Python 3 only, which is simpler than trying to support both.
    As a result, support for Python 2.7 in this example project has been dropped.

Windows Support
---------------

Summary: On Windows, use ``py`` instead of ``python3`` for many of the examples in this
documentation.

This package fully supports Windows, along with Linux and macOS, but Python is typically
`installed differently on Windows <https://docs.python.org/3/using/windows.html>`_.
Windows users typically access Python through the
`py <https://www.python.org/dev/peps/pep-0397/>`_ launcher rather than a ``python3``
link in their ``PATH``. Within a virtual environment, all platforms operate the same and use a
``python`` link to access the Python version used in that virtual environment.

Dependencies
------------

Dependencies are defined in:

- ``requirements.in``

- ``requirements.txt``

- ``dev-requirements.in``

- ``dev-requirements.txt``

Virtual Environments
^^^^^^^^^^^^^^^^^^^^

It is best practice during development to create an isolated
`Python virtual environment <https://docs.python.org/3/library/venv.html>`_ using the
``venv`` standard library module. This will keep dependant Python packages from interfering
with other Python projects on your system.

On \*Nix:

.. code-block:: bash

    # On Python 3.9+, add --upgrade-deps
    $ python3 -m venv venv
    $ source venv/bin/activate

On Windows ``cmd``:

.. code-block:: bash

    > py -m venv venv
    > venv\Scripts\activate.bat

Once activated, it is good practice to update core packaging tools (``pip``, ``setuptools``, and
``wheel``) to the latest versions.

.. code-block:: bash

    (venv) $ python -m pip install --upgrade pip setuptools wheel

