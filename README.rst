kot
===

A toy programing language.

.. image:: https://travis-ci.org/jstasiak/kotlang.svg?branch=master
    :target: https://travis-ci.org/jstasiak/kotlang

Requirements
------------

* Mac OS or Linux
* libclang 7 development library
* A cc binary that can link things
* Python 3.7+

Installation on Mac OS with HomeBrew::

    brew update && brew upgrade && install llvm python3

Architecture
------------

* The parser is written by hand, writing parsers is good fun
* Everything connected to code generation is handled by llvmlite, great library
* libclang is used to parse C headers (libclang Python bindings are vendored
  by this project to not depend on them being installed separately, tricky on Linux)

Running tests
-------------

The run tests (assumes activated Python 3.7 virtualenv and the current directory
being a cloned repository root)::

  pip install --upgrade '.[dev]'
  make ci
