# :microscope: How to test NARPS open pipelines ?

:mega: This file descripes the test suite and features for the project.

## Dependancies

Use [*pylint*](http://pylint.pycqa.org/en/latest/) to run static code analysis.

> Pylint is a tool that checks for errors in Python code, tries to enforce a coding standard and looks for code smells. It can also look for certain type errors, it can recommend suggestions about how particular blocks can be refactored and can offer you details about the code's complexity.

* Run the analysis on all the source files with : `pylint ./narps_open`
* To create a .xml JUnit report : `pylint --fail-under=8.0 --ignored-classes=_socketobject --load-plugins=pylint_junit --output-format=pylint_junit.JUnitReporter narps_open > pylint_report_narps_open.xml`

It is also a good idea to use [*black*](https://github.com/psf/black) to automatically conform your code to PEP8.

> Black is the uncompromising Python code formatter. By using it, you agree to cede control over minutiae of hand-formatting. In return, Black gives you speed, determinism, and freedom from pycodestyle nagging about formatting. You will save time and mental energy for more important matters.

* Run the command on any source file you want to lint : `black ./narps_open` or `black ./narps_open/runner.py` 

Use [*pytest*](https://docs.pytest.org/en/6.2.x/contents.html) to run automatic testing and its [*pytest-cov*](https://pytest-cov.readthedocs.io/en/latest/) plugin to control code coverage. Furthermore, [*pytest-helpers-namespace*](https://pypi.org/project/pytest-helpers-namespace/) enables to register helper functions.

> The pytest framework makes it easy to write small tests, yet scales to support complex functional testing for applications and libraries.

## Launching tests

Tests can be launched manually or while using CI (Continuous Integration).

* To run the tests : `pytest ./tests` or `pytest`
* To specify a test file to run : `pytest test_file.py`
* To specify a test -for which the name contains 'test_pattern'- inside a test file : `pytest test_file.py -k "test_pattern"`
* To run a tests with a given mark 'mark' : `pytest -m 'mark'`
* [CI] to output a xml JUnit test report, use the option : `--junit-xml=pytest_report.xml`
* To create code coverage data : `coverage run -m pytest ./tests` then `coverage report` to see the code coverage result or `coverage xml` to output a .xml report file

## Configuration files for testing

* `pytest.ini` is a global configuration files for using pytest (see reference [here](https://docs.pytest.org/en/7.1.x/reference/customize.html)). It allows to [register markers](https://docs.pytest.org/en/7.1.x/example/markers.html) that help to better identify tests.
* `tests/conftest.py` defines common functions, parameters, and [helpers](https://pytest-helpers-namespace.readthedocs.io/en/latest/) that are later available to all tests

## Types of tests

We use pytest [markers](https://docs.pytest.org/en/7.1.x/example/markers.html) to identify the different types of test. Currently, the following types are available :

| Type of test | marker | Description |
| ----------- | ----------- | ----------- |
| unit tests | `unit_test` | Unitary test a method/function |
| pipeline tests | `pieline_test` | These tests compute whole pipeline one or several times making them time and resources consuming. |

## Writing tests

The main idea is to create one test file per source module (eg.: *tests/pipelines/test_pipelines.py* contains all the unit tests for the module `narps_open.pipelines`).

Each test file defines a class (in the example: `TestPipelines`), in which each test is written in a static method begining with `test_`.

Finally we use one or several `assert` ; each one of them making the whole test fail if the assertion is False. One can also use the `raises` method of pytest, writing `with raises(Exception):` to test if a piece of code raised the expected Exception. See the reference [here](https://docs.pytest.org/en/6.2.x/reference.html?highlight=raises#pytest.raises).

## Non regression testing

Here is a procedure on how to perform a regression test, when modifying the code of a pipeline. In the following we test pipeline 2T6S.

1. Checkout commit c70e820 and launch jupyter inside a docker container

```bash
# from inside your repository
git checkout c70e820

# run a docker container
docker run -it --rm -v <path/to/your/narps_open_pipelines/repository>:/home/neuro/code/ -v <path/to/the/dataset/ds001734>:/data/ -v <path/to/the/output/directory/c70e820>:/output/ -p 8888:8888 elodiegermani/open_pipeline

# from inside the container
jupyter notebook --port=8888 --no-browser --ip=0.0.0.0

    To access the notebook, open this file in a browser:
        file:///home/neuro/.local/share/jupyter/runtime/nbserver-17-open.html
    Or copy and paste one of these URLs:
        http://51def897c243:8888/?token=02856fdbf3dbf8b382e28381707b64ff8650cf41b6ec67d8
     or http://127.0.0.1:8888/?token=02856fdbf3dbf8b382e28381707b64ff8650cf41b6ec67d8
```

2. Open one of the URLs provided by jupyter, open `src/reproduction_2T6S.ipynb`, change the following parameters and run the notebook.

```python
exp_dir = '/data/'
result_dir = '/output/'
subject_list=['001', '002', '003', '004']
```

3. Check the results.

```bash
cd <path/to/the/output/directory/c70e820>
tree NARPS-2T6S-reproduced/l2_analysis_* -P *.nii
    NARPS-2T6S-reproduced/l2_analysis_equalIndifference_nsub_4
    ├── _contrast_id_01
    │      ├── con_0001.nii
    │      ├── con_0002.nii
    │      ├── mask.nii
    │      ├── spmT_0001.nii
    │      ├── spmT_0002.nii
    │      ├── _threshold0
    │      │      └── spmT_0001_thr.nii
    │      └── _threshold1
    │          └── spmT_0002_thr.nii
    ├── _contrast_id_02
    │      ├── con_0001.nii
    │      ├── con_0002.nii
    │      ├── mask.nii
    │      ├── spmT_0001.nii
    │      ├── spmT_0002.nii
    │      ├── _threshold0
    │      │      └── spmT_0001_thr.nii
    │      └── _threshold1
    │          └── spmT_0002_thr.nii
    ├── _contrast_id_03
    │      ├── con_0001.nii
    │      ├── con_0002.nii
    │      ├── mask.nii
    │      ├── spmT_0001.nii
    │      ├── spmT_0002.nii
    │      ├── _threshold0
    │      │      └── spmT_0001_thr.nii
    │      └── _threshold1
    │          └── spmT_0002_thr.nii
    └── _contrast_id_04
        ├── con_0001.nii
        ├── con_0002.nii
        ├── mask.nii
        ├── spmT_0001.nii
        ├── spmT_0002.nii
        ├── _threshold0
        │      └── spmT_0001_thr.nii
        └── _threshold1
            └── spmT_0002_thr.nii
    NARPS-2T6S-reproduced/l2_analysis_equalRange_nsub_4
    ├── _contrast_id_01
    │      ├── con_0001.nii
    │      ├── con_0002.nii
    │      ├── mask.nii
    │      ├── spmT_0001.nii
    │      ├── spmT_0002.nii
    │      ├── _threshold0
    │      │      └── spmT_0001_thr.nii
    │      └── _threshold1
    │          └── spmT_0002_thr.nii
    ├── _contrast_id_02
    │      ├── con_0001.nii
    │      ├── con_0002.nii
    │      ├── mask.nii
    │      ├── spmT_0001.nii
    │      ├── spmT_0002.nii
    │      ├── _threshold0
    │      │      └── spmT_0001_thr.nii
    │      └── _threshold1
    │          └── spmT_0002_thr.nii
    ├── _contrast_id_03
    │      ├── con_0001.nii
    │      ├── con_0002.nii
    │      ├── mask.nii
    │      ├── spmT_0001.nii
    │      ├── spmT_0002.nii
    │      ├── _threshold0
    │      │      └── spmT_0001_thr.nii
    │      └── _threshold1
    │          └── spmT_0002_thr.nii
    └── _contrast_id_04
        ├── con_0001.nii
        ├── con_0002.nii
        ├── mask.nii
        ├── spmT_0001.nii
        ├── spmT_0002.nii
        ├── _threshold0
        │      └── spmT_0001_thr.nii
        └── _threshold1
            └── spmT_0002_thr.nii
    NARPS-2T6S-reproduced/l2_analysis_groupComp_nsub_4
    ├── _contrast_id_01
    │      ├── con_0001.nii
    │      ├── mask.nii
    │      ├── spmT_0001.nii
    │      └── _threshold0
    │          └── spmT_0001_thr.nii
    ├── _contrast_id_02
    │      ├── con_0001.nii
    │      ├── mask.nii
    │      ├── spmT_0001.nii
    │      └── _threshold0
    │          └── spmT_0001_thr.nii
    ├── _contrast_id_03
    │      ├── con_0001.nii
    │      ├── mask.nii
    │      ├── spmT_0001.nii
    │      └── _threshold0
    │          └── spmT_0001_thr.nii
    └── _contrast_id_04
        ├── con_0001.nii
        ├── mask.nii
        ├── spmT_0001.nii
        └── _threshold0
            └── spmT_0001_thr.nii
```

4. Switch back to your current development branch (code revision to be tested), and run the pipeline. :warning: Choose a different `<path/to/the/output/directory/dev>` in the docker command line, so that the results from the previous run are not replaced.

```bash
# from inside your repository
git switch <your_branch_name>

# run a docker container
docker run -it --rm -v <path/to/your/narps_open_pipelines/repository>:/home/neuro/code/ -v <path/to/the/dataset/ds001734>:/data/ -v <path/to/the/output/directory/dev>:/output/ elodiegermani/open_pipeline

# from inside the container
cd /home/neuro/code
source activate neuro
pip install .
python narps_open/runner.py -t 2T6S -s 1 2 3 4 -d /data/ -o /output/

# leave the container
exit
```

5. At this point, you can check the results (as in step 3)

```bash
cd <path/to/the/output/directory/dev>
tree NARPS-2T6S-reproduced/l2_analysis_* -P *.nii

...
```

6. Open a container again, to compare the results

```bash
# run a docker container
docker run -it --rm -v <path/to/your/narps_open_pipelines/repository>:/home/neuro/code/ -v <path/to/the/output/directory/c70e820>:/output_1/ -v <path/to/the/output/directory/dev>:/output_2/ elodiegermani/open_pipeline

# from inside the container
cd /home/neuro/code
source activate neuro
pip install .
```

7. Launch the following python code from inside the container

```python
from narps_open.utils.correlation import get_correlation_coefficient

output_files_c70e820 = [
    '/output_1/NARPS-2T6S-reproduced/l2_analysis_equalIndifference_nsub_4/_contrast_id_01/con_0001.nii',
    '/output_1/NARPS-2T6S-reproduced/l2_analysis_equalIndifference_nsub_4/_contrast_id_01/con_0002.nii',
    '/output_1/NARPS-2T6S-reproduced/l2_analysis_equalIndifference_nsub_4/_contrast_id_02/con_0001.nii',
    '/output_1/NARPS-2T6S-reproduced/l2_analysis_equalIndifference_nsub_4/_contrast_id_02/con_0002.nii',
    '/output_1/NARPS-2T6S-reproduced/l2_analysis_equalIndifference_nsub_4/_contrast_id_03/con_0001.nii',
    '/output_1/NARPS-2T6S-reproduced/l2_analysis_equalIndifference_nsub_4/_contrast_id_03/con_0002.nii',
    '/output_1/NARPS-2T6S-reproduced/l2_analysis_equalIndifference_nsub_4/_contrast_id_04/con_0001.nii',
    '/output_1/NARPS-2T6S-reproduced/l2_analysis_equalIndifference_nsub_4/_contrast_id_04/con_0002.nii',
    '/output_1/NARPS-2T6S-reproduced/l2_analysis_equalRange_nsub_4/_contrast_id_01/con_0001.nii',
    '/output_1/NARPS-2T6S-reproduced/l2_analysis_equalRange_nsub_4/_contrast_id_01/con_0002.nii',
    '/output_1/NARPS-2T6S-reproduced/l2_analysis_equalRange_nsub_4/_contrast_id_02/con_0001.nii',
    '/output_1/NARPS-2T6S-reproduced/l2_analysis_equalRange_nsub_4/_contrast_id_02/con_0002.nii',
    '/output_1/NARPS-2T6S-reproduced/l2_analysis_equalRange_nsub_4/_contrast_id_03/con_0001.nii',
    '/output_1/NARPS-2T6S-reproduced/l2_analysis_equalRange_nsub_4/_contrast_id_03/con_0002.nii',
    '/output_1/NARPS-2T6S-reproduced/l2_analysis_equalRange_nsub_4/_contrast_id_04/con_0001.nii',
    '/output_1/NARPS-2T6S-reproduced/l2_analysis_equalRange_nsub_4/_contrast_id_04/con_0002.nii',
    '/output_1/NARPS-2T6S-reproduced/l2_analysis_groupComp_nsub_4/_contrast_id_01/con_0001.nii',
    '/output_1/NARPS-2T6S-reproduced/l2_analysis_groupComp_nsub_4/_contrast_id_02/con_0001.nii',
    '/output_1/NARPS-2T6S-reproduced/l2_analysis_groupComp_nsub_4/_contrast_id_03/con_0001.nii',
    '/output_1/NARPS-2T6S-reproduced/l2_analysis_groupComp_nsub_4/_contrast_id_04/con_0001.nii'
]

output_files_dev = [
    '/output_2/NARPS-2T6S-reproduced/l2_analysis_equalIndifference_nsub_4/_contrast_id_0001/con_0001.nii',
    '/output_2/NARPS-2T6S-reproduced/l2_analysis_equalIndifference_nsub_4/_contrast_id_0001/con_0002.nii',
    '/output_2/NARPS-2T6S-reproduced/l2_analysis_equalIndifference_nsub_4/_contrast_id_0002/con_0001.nii',
    '/output_2/NARPS-2T6S-reproduced/l2_analysis_equalIndifference_nsub_4/_contrast_id_0002/con_0002.nii',
    '/output_2/NARPS-2T6S-reproduced/l2_analysis_equalIndifference_nsub_4/_contrast_id_0003/con_0001.nii',
    '/output_2/NARPS-2T6S-reproduced/l2_analysis_equalIndifference_nsub_4/_contrast_id_0003/con_0002.nii',
    '/output_2/NARPS-2T6S-reproduced/l2_analysis_equalIndifference_nsub_4/_contrast_id_0004/con_0001.nii',
    '/output_2/NARPS-2T6S-reproduced/l2_analysis_equalIndifference_nsub_4/_contrast_id_0004/con_0002.nii',
    '/output_2/NARPS-2T6S-reproduced/l2_analysis_equalRange_nsub_4/_contrast_id_0001/con_0001.nii',
    '/output_2/NARPS-2T6S-reproduced/l2_analysis_equalRange_nsub_4/_contrast_id_0001/con_0002.nii',
    '/output_2/NARPS-2T6S-reproduced/l2_analysis_equalRange_nsub_4/_contrast_id_0002/con_0001.nii',
    '/output_2/NARPS-2T6S-reproduced/l2_analysis_equalRange_nsub_4/_contrast_id_0002/con_0002.nii',
    '/output_2/NARPS-2T6S-reproduced/l2_analysis_equalRange_nsub_4/_contrast_id_0003/con_0001.nii',
    '/output_2/NARPS-2T6S-reproduced/l2_analysis_equalRange_nsub_4/_contrast_id_0003/con_0002.nii',
    '/output_2/NARPS-2T6S-reproduced/l2_analysis_equalRange_nsub_4/_contrast_id_0004/con_0001.nii',
    '/output_2/NARPS-2T6S-reproduced/l2_analysis_equalRange_nsub_4/_contrast_id_0004/con_0002.nii',
    '/output_2/NARPS-2T6S-reproduced/l2_analysis_groupComp_nsub_4/_contrast_id_0001/con_0001.nii',
    '/output_2/NARPS-2T6S-reproduced/l2_analysis_groupComp_nsub_4/_contrast_id_0002/con_0001.nii',
    '/output_2/NARPS-2T6S-reproduced/l2_analysis_groupComp_nsub_4/_contrast_id_0003/con_0001.nii',
    '/output_2/NARPS-2T6S-reproduced/l2_analysis_groupComp_nsub_4/_contrast_id_0004/con_0001.nii'
]

for file_1, file_2 in zip(output_files_c70e820, output_files_dev):
    print(get_correlation_coefficient(file_1, file_2))
```
