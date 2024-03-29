cqed_sims
================

.. image:: https://github.com/johnthagen/python-blueprint/workflows/python/badge.svg
    :target: https://github.com/melmatt7/cqed_photon_statistics/actions

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://black.readthedocs.io/en/stable/

.. image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
    :target: https://timothycrosley.github.io/isort/

cavity quantum electrodynamic system simulations

Installation
---------------

This package fully supports Windows, along with Linux and macOS, but Python is typically
`installed differently on Windows <https://docs.python.org/3/using/windows.html>`_.
Windows users typically access Python through the
`py <https://www.python.org/dev/peps/pep-0397/>`_ launcher rather than a ``python3``
link in their ``PATH``. Within a virtual environment,  or when using conda, all platforms operate the same and use a
``python`` link to access the Python version used in that virtual environment.

Requirements
^^^^^^^^^^^^

Python 3.6+.

Dependencies
^^^^^^^^^^^^

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

Locking Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This project uses `pip-tools <https://github.com/jazzband/pip-tools>`_ to lock project
dependencies and create reproducible virtual environments.

**Note:** *Library* projects should not lock their ``requirements.txt``. Since ``python-blueprint``
also has a CLI application, this end-user application example is used to demonstrate how to
lock application dependencies.

To update dependencies:

.. code-block:: bash

    (venv) $ python -m pip install pip-tools
    (venv) $ python -m piptools compile --upgrade requirements.in
    (venv) $ python -m piptools compile --upgrade dev-requirements.in

After upgrading dependencies, run the unit tests as described in the `Unit Testing`_ section
to ensure that none of the updated packages caused incompatibilities in the current project.

Syncing Virtual Environments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To cleanly install your dependencies into your virtual environment:

.. code-block:: bash

    (venv) $ python -m piptools sync requirements.txt dev-requirements.txt
