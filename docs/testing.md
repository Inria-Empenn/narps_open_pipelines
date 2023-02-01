# How to test NARPS open pipelines ?

:mega: This file descripes the test suite and features for the project.

## Dependancies

Use *pylint* to run static code analysis.

*[Pylint](http://pylint.pycqa.org/en/latest/) is a tool that checks for errors in Python code, tries to enforce a coding standard and looks for code smells. It can also look for certain type errors, it can recommend suggestions about how particular blocks can be refactored and can offer you details about the code's complexity.*

* Run the analysis on all the source files with : `pylint ./narps_open`
* To create a .xml JUnit report : `pylint --fail-under=8.0 --ignored-classes=_socketobject --load-plugins=pylint_junit --output-format=pylint_junit.JUnitReporter narps_open > pylint_report_narps_open.xml`

Use [*pytest*](https://docs.pytest.org/en/6.2.x/contents.html) to run automatic testing and its [*pytest-cov*](https://pytest-cov.readthedocs.io/en/latest/) plugin to control code coverage.

*The pytest framework makes it easy to write small tests, yet scales to support complex functional testing for applications and libraries.*

Tests can be launched manually or while using CI (Continuous Integration).

* To run the tests : `pytest ./tests`
* To specify a test file to run : `pytest test_file.py`
* To specify a test -for which the name contains 'test_pattern'- inside a test file : `pytest test_file.py -k "test_pattern"`
* [CI] to output a xml JUnit test report, use the option : `--junit-xml=pytest_report.xml`
* To create code coverage data : `coverage run -m pytest ./tests` then `coverage report` to see the code coverage result or `coverage xml` to output a .xml report file

## Writing tests
The main idea is to create one test file per source module (eg.: *tests/pipelines/test_pipelines.py* contains all the tests for the module `narps_open.pipelines`).

Each test file defines a class (in the example: `TestPipelines`), in which each test is written in a static method begining with `test_`.

Finally we use one or several `assert` ; each one of them making the whole test fail if the assertion is False. One can also use the `raises` method of pytest, writing `with raises(Exception):` to test if a piece of code raised the expected Exception. See the reference [here](https://docs.pytest.org/en/6.2.x/reference.html?highlight=raises#pytest.raises).
