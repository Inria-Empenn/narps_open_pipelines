#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_C88N' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_C88N.py
    pytest -q test_team_C88N.py -k <selected_test>
"""

from statistics import mean

from pytest import raises, helpers, mark
from nipype import Workflow

from narps_open.utils.configuration import Configuration
from narps_open.pipelines.team_C88N import PipelineTeamC88N

class TestPipelinesTeamC88N:
    """ A class that contains all the unit tests for the PipelineTeamC88N class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineTeamC88N object """

        pipeline = PipelineTeamC88N()

        # 1 - check the parameters
        assert pipeline.fwhm == 8.0
        assert pipeline.team_id == 'C88N'

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
        """ Test the execution of a PipelineTeamC88N and compare results """
        results_4_subjects = helpers.test_pipeline(
            'C88N',
            Configuration()['directories']['narps_results'],
            Configuration()['directories']['dataset'],
            Configuration()['directories']['reproduced_results'],
            4)
        assert mean(results_4_subjects) > .003
