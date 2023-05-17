#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_2T6S' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_2T6S.py
    pytest -q test_team_2T6S.py -k <selected_test>
"""

from statistics import mean

from pytest import helpers, mark
from nipype import Workflow

from narps_open.pipelines.team_2T6S import PipelineTeam2T6S

class TestPipelinesTeam2T6S:
    """ A class that contains all the unit tests for the PipelineTeam2T6S class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineTeam2T6S object """

        pipeline = PipelineTeam2T6S()

        # 1 - check the parameters
        assert pipeline.fwhm == 8.0
        assert pipeline.team_id == '2T6S'

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
        """ Test the expected outputs of a PipelineTeam2T6S object """
        pipeline = PipelineTeam2T6S()
        # 1 - 1 suject outputs
        pipeline.subject_list = ['001']
        assert len(pipeline.get_preprocessing_outputs()) == 0
        assert len(pipeline.get_run_level_outputs()) == 0
        assert len(pipeline.get_subject_level_outputs()) == 9
        assert len(pipeline.get_group_level_outputs()) == 84

        # 2 - 4 sujects outputs
        pipeline.subject_list = ['001', '002', '003', '004']
        assert len(pipeline.get_preprocessing_outputs()) == 0
        assert len(pipeline.get_run_level_outputs()) == 0
        assert len(pipeline.get_subject_level_outputs()) == 36
        assert len(pipeline.get_group_level_outputs()) == 84

    @staticmethod
    @mark.pipeline_test
    def test_execution():
        """ Test the execution of a PipelineTeam2T6S and compare results """
        results_4_subjects = helpers.test_pipeline(
            '2T6S',
            '/references/',
            '/data/',
            '/output/',
            4)
        assert mean(results_4_subjects) > .003
