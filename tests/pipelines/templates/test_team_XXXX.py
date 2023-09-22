#!/usr/bin/python
# coding: utf-8

""" This template can be use to test a pipeline.

    - Replace all occurrences of XXXX by the actual id of the team.
    - All lines starting with [INFO], are meant to help you during the reproduction,
    these can be removed eventually.
    - Also remove lines starting with [TODO], once you did what they suggested.
    - Remove this docstring once you are done with coding the tests.
"""

""" Tests of the 'narps_open.pipelines.team_XXXX' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_XXXX.py
    pytest -q test_team_XXXX.py -k <selected_test>
"""

# [INFO] About these imports :
# [INFO]      - pytest.helpers allows to use the helpers registered in tests/conftest.py
# [INFO]      - pytest.mark allows to categorize tests as unitary or pipeline tests
from pytest import helpers, mark

# [INFO] Only for type testing
from nipype import Workflow

# [INFO] Of course, import the class you want to test, here the Pipeline class for the team XXXX
from narps_open.pipelines.team_XXXX import PipelineTeamXXXX

# [INFO] All tests should be contained in the following class, in order to sort them.
class TestPipelinesTeamXXXX:
    """ A class that contains all the unit tests for the PipelineTeamXXXX class."""

    # [TODO] Write one or several unit_test (and mark them as such)
    # [TODO]    ideally for each method of the class you test.

    # [INFO] Here is one example for the __init__() method
    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineTeamXXXX object """

        pipeline = PipelineTeamXXXX()
        assert pipeline.fwhm == 8.0
        assert pipeline.team_id == 'XXXX'

    # [INFO] Here is one example for the methods returning workflows
    @staticmethod
    @mark.unit_test
    def test_workflows():
        """ Test the workflows of a PipelineTeamXXXX object """

        pipeline = PipelineTeamXXXX()
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
        """ Test the expected outputs of a PipelineTeamXXXX object """
        pipeline = PipelineTeamXXXX()

        # 1 - 1 subject outputs
        pipeline.subject_list = ['001']
        assert len(pipeline.get_preprocessing_outputs()) == 0
        assert len(pipeline.get_run_level_outputs()) == 0
        assert len(pipeline.get_subject_level_outputs()) == 7
        assert len(pipeline.get_group_level_outputs()) == 63
        assert len(pipeline.get_hypotheses_outputs()) == 18

        # 2 - 4 subjects outputs
        pipeline.subject_list = ['001', '002', '003', '004']
        assert len(pipeline.get_preprocessing_outputs()) == 0
        assert len(pipeline.get_run_level_outputs()) == 0
        assert len(pipeline.get_subject_level_outputs()) == 28
        assert len(pipeline.get_group_level_outputs()) == 63
        assert len(pipeline.get_hypotheses_outputs()) == 18

    # [TODO] Feel free to add other methods, e.g. to test the custom node functions of the pipeline

    # [TODO] Write one pipeline_test (and mark it as such)

    # [INFO] The pipeline_test will most likely be exactly written this way :
    @staticmethod
    @mark.pipeline_test
    def test_execution():
        """ Test the execution of a PipelineTeamXXXX and compare results """

        # [INFO] We use the `test_pipeline_evaluation` helper which is responsible for running the
        # [INFO]    pipeline, iterating over subjects and comparing output with expected results.
        helpers.test_pipeline_evaluation('XXXX')
