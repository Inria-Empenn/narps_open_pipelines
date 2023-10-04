#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_98BT' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_98BT.py
    pytest -q test_team_98BT.py -k <selected_test>
"""

from pytest import helpers, mark
from nipype import Workflow

from narps_open.pipelines.team_98BT import PipelineTeam98BT

class TestPipelinesTeam98BT:
    """ A class that contains all the unit tests for the PipelineTeam98BT class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineTeam98BT object """

        pipeline = PipelineTeam98BT()

        # 1 - check the parameters
        assert pipeline.fwhm == 8.0
        assert pipeline.team_id == '98BT'

        # 2 - check workflows
        processing = pipeline.get_preprocessing()
        assert len(processing) == 2
        for sub_workflow in processing:
            assert isinstance(sub_workflow, Workflow)

        assert pipeline.get_run_level_analysis() is None
        assert isinstance(pipeline.get_subject_level_analysis(), Workflow)

        group_level = pipeline.get_group_level_analysis()
        assert len(group_level) == 3
        for sub_workflow in group_level:
            assert isinstance(sub_workflow, Workflow)

    @staticmethod
    @mark.unit_test
    def test_outputs():
        """ Test the expected outputs of a PipelineTeam98BT object """
        pipeline = PipelineTeam98BT()
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
        """ Test the execution of a PipelineTeam98BT and compare results """
        helpers.test_pipeline_evaluation('98BT')
