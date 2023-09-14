#######################
summit_testing_analysis
#######################

``summit_testing_analysis`` is a package in the `LSST Science Pipelines <https://pipelines.lsst.io>`_.

It contains code to analyse the performance of Rubin Observatory's components. 
Such as M1M3, M2, hexapods, rotator, mount, etc. 

Contributing
============

* This repository uses the development guidelines described in
  https://developer.lsst.io/.

* The code format should follow the guidelines in 
  https://developer.lsst.io/python/formatting.html

  We use two tools to help with code standards compliance. 
  They are `black` and `isort`. Use them with the following commands:

.. code-block:: bash

    black .
    isort .

* This repository is configure with a pre-commit hook. In other words, 
  it contains an automatic code check before you commit your changes. 
  Before you can commit your changes, you need to install the pre-commit:

.. code-block:: bash

    mamba install -c lsstts ts-pre-commit-config
    generate_pre_commit_conf
    pre-commit install 
    pre-commit run --all-files

