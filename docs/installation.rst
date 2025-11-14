.. _installation:

Installation
============

Getting started
---------------

The best practice to follow for using/contributing to the package are as follows:

1. Fork the project `repository <https://gitlab.desy.de/itsaklid/sysvar>`_: Click on the *Fork* button near the top of the
   page. This creates a copy of the code under your account.

2. Clone this copy to your local disk:

   .. code-block:: bash

       git clone git@github.com:YourLogin/SysVar.git
       cd SysVar

3. Now, you are ready to install and use the package with a simple:

   For developer mode:
   
   .. code-block:: bash

       pip install -e .[dev]

   For end-users (not recommended currently):
   
   .. code-block:: bash

       pip install .

4. (Optional: for developers) Install ``pre-commit``:

   .. code-block:: bash

       pip install pre-commit
       pre-commit install

   This ensures that the code being written is not susceptible to any trivial styling issues.

Setting the SysVar Path
-----------------------

**If you run into any difficulties while accesssing the package in your scripts,please add the following two lines at the very beginning of every script that uses SysVar to point to the location where you have installed your SysVar fork**

.. code-block:: python

    import sys
    sys.path.insert(0,'{path_where_you_pip_installed_sysvar}/SysVar/src')


