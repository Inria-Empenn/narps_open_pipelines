#!/usr/bin/python
# coding: utf-8

""" This template can be use to test a pipeline.

    - Replace all occurrences of 2T6S by the actual id of the team.
    - All lines starting with [INFO], are meant to help you during the reproduction,
    these can be removed eventually.
    - Also remove lines starting with [TODO], once you did what they suggested.
    - Remove this docstring once you are done with coding the tests.
"""

""" Tests of the 'narps_open.pipelines.team_2T6S' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_2T6S.py
    pytest -q test_team_2T6S.py -k <selected_test>
"""

# [INFO] About these imports :
# [INFO]      - pytest.helpers allows to use the helpers registered in tests/conftest.py
# [INFO]      - pytest.mark allows to categorize tests as unitary or pipeline tests
from pytest import helpers, mark

# [INFO] Only for type testing
from nipype import Workflow

# [INFO] Of course, import the class you want to test, here the Pipeline class for the team 2T6S
from narps_open.pipelines.team_2T6S import PipelineTeam2T6S

# [INFO] All tests should be contained in the following class, in order to sort them.
class TestPipelinesTeam2T6S:
    """ A class that contains all the unit tests for the PipelineTeam2T6S class."""

    # [TODO] Write one or several unit_test (and mark them as such)
    # [TODO]    ideally for each method of the class you test.

    # [INFO] Here is one example for the __init__() method
    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineTeam2T6S object """

        pipeline = PipelineTeam2T6S()
        assert pipeline.fwhm == 8.0
        assert pipeline.team_id == '2T6S'

    # [INFO] Here is one example for the methods returning workflows
    @staticmethod
    @mark.unit_test
    def test_workflows():
        """ Test the workflows of a PipelineTeam2T6S object """

        pipeline = PipelineTeam2T6S()
        assert pipeline.get_preprocessing() is None
        assert pipeline.get_run_level_analysis() is None
        assert isinstance(pipeline.get_subject_level_analysis(), Workflow)
        group_level = pipeline.get_group_level_analysis()

        assert len(group_level) == 3
        for sub_workflow in group_level:
            assert isinstance(sub_workflow, Workflow)

    # [INFO] Here is one example for the methods returning outputs
    @staticmethod
    @mark.unit_test
    def test_outputs():
        """ Test the expected outputs of a PipelineTeam2T6S object """
        pipeline = PipelineTeam2T6S()

        # 1 - 1 subject outputs
        pipeline.subject_list = ['001']
        helpers.test_pipeline_outputs(pipeline, [0, 0, 7, 63, 18])

        # 2 - 4 subjects outputs
        pipeline.subject_list = ['001', '002', '003', '004']
        helpers.test_pipeline_outputs(pipeline, [0, 0, 28, 63, 18])

    # [TODO] Feel free to add other methods, e.g. to test the custom node functions of the pipeline

    # [TODO] Write one pipeline_test (and mark it as such)

    # [INFO] The pipeline_test will most likely be exactly written this way :
    @staticmethod
    @mark.pipeline_test
    def test_execution():
        """ Test the execution of a PipelineTeam2T6S and compare results """

        # [INFO] We use the `test_pipeline_evaluation` helper which is responsible for running the
        # [INFO]    pipeline, iterating over subjects and comparing output with expected results.
        helpers.test_pipeline_evaluation('2T6S')
