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

from pytest import raises, helpers, mark
from nipype import Workflow

from narps_open.pipelines.team_2T6S import PipelineTeam2T6S

class TestPipelinesTeam2T6S:
    """ A class that contains all the unit tests for the PipelineTeam2T6S class."""

    @staticmethod
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
