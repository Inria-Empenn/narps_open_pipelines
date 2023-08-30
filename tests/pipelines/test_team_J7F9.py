#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_J7F9' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_J7F9.py
    pytest -q test_team_J7F9.py -k <selected_test>
"""

from pytest import helpers, mark
from nipype import Workflow

from narps_open.pipelines.team_J7F9 import PipelineTeamJ7F9

class TestPipelinesTeamJ7F9:
    """ A class that contains all the unit tests for the PipelineTeamJ7F9 class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineTeamJ7F9 object """

        pipeline = PipelineTeamJ7F9()

        # 1 - check the parameters
        assert pipeline.fwhm == 8.0
        assert pipeline.team_id == 'J7F9'

        # 2 - check workflows
        assert pipeline.get_preprocessing() is None
        assert pipeline.get_run_level_analysis() is None
        assert isinstance(pipeline.get_subject_level_analysis(), Workflow)
        group_level = pipeline.get_group_level_analysis()

        assert len(group_level) == 3
        for sub_workflow in group_level:
            assert isinstance(sub_workflow, Workflow)

    @staticmethod
    @mark.unit_test
    def test_outputs():
        """ Test the expected outputs of a PipelineTeamJ7F9 object """
        pipeline = PipelineTeamJ7F9()
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

    @staticmethod
    @mark.pipeline_test
    def test_execution():
        """ Test the execution of a PipelineTeamJ7F9 and compare results """
        helpers.test_pipeline_evaluation('J7F9')
