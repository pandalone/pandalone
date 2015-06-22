###################
Contributing
###################
.. _start:

For submitting code, use ``UTF-8`` everywhere, unix-eol(``LF``) and 
set ``git --config core.autocrlf = input``.

The typical development procedure is like this:

1. Modify the sources in small, isolated and well-defined changes, i.e.
   adding a single feature, or fixing a specific bug.

2. Add test-cases "proving" your code.

3. Rerun all test-cases to ensure that you didn't break anything,
   and check their *coverage* remain above 80%:

    .. code-block:: console

        $ python setup.py test_code_cover


    .. Tip:: You can enter just: ``python setup.py test_all`` instead of 
        the above cmd-line since it has been *aliased* in the `setup.cfg` file.
        Check this file for more example commands to use during development.


4. To see the rendered results of the documents, issue the following commands 
   and read the result html at `build/sphinx/html/index.html`:

    .. code-block:: console

        $ python setup.py build_sphinx                  # Builds html docs
        $ python setup.py build_sphinx -b doctest       # Checks if python-code embeded in comments runs ok.


5. If there are no problems, commit your changes with a descriptive message.

6. Repeat this cycle for further modifications.

7. If you made a rather important modification, update also documentation 
   (i.e. README.rst) the `CHANGES` and `AUTHORS`.

8. When you are finished, push the changes upstream to *github* and 
   make a *merge_request*. You can check whether your merge-request indeed 
   passed the tests by checking its build-status, on both integration site:
   Travis: |travis-status|, Appveyor: |appveyor-status|.

.. Hint:: 
    Skim through these small guides:

    - `IPython developer's documentantion: The perfect pull request 
      <https://github.com/ipython/ipython/wiki/Dev:-The-perfect-pull-request>`_

    - `Effective pull requests and other good practices for teams using github 
      <http://codeinthehole.com/writing/pull-requests-and-other-good-practices-for-teams-using-github/>`_


