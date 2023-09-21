#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_3C6G' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_3C6G.py
    pytest -q test_team_3C6G.py -k <selected_test>
"""

from pytest import helpers, mark
from nipype import Workflow

from narps_open.pipelines.team_3C6G import PipelineTeam3C6G

class TestPipelinesTeam3C6G:
    """ A class that contains all the unit tests for the PipelineTeam3C6G class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineTeam3C6G object """

        pipeline = PipelineTeam3C6G()

        # 1 - check the parameters
        assert pipeline.fwhm == 6.0
        assert pipeline.team_id == '3C6G'

        # 2 - check workflows
        assert isinstance(pipeline.get_preprocessing(), Workflow)
        assert pipeline.get_run_level_analysis() is None
        assert isinstance(pipeline.get_subject_level_analysis(), Workflow)
        group_level = pipeline.get_group_level_analysis()

        assert len(group_level) == 3
        for sub_workflow in group_level:
            assert isinstance(sub_workflow, Workflow)

    @staticmethod
    @mark.unit_test
    def test_outputs():
        """ Test the expected outputs of a PipelineTeam3C6G object """
        pipeline = PipelineTeam3C6G()
        # 1 - 1 subject - 1 run outputs
        pipeline.subject_list = ['001']
        pipeline.run_list = ['01']
        assert len(pipeline.get_preprocessing_outputs()) == 14
        assert len(pipeline.get_run_level_outputs()) == 0
        assert len(pipeline.get_subject_level_outputs()) == 7
        #assert len(pipeline.get_group_level_outputs()) == 63
        #assert len(pipeline.get_hypotheses_outputs()) == 18

        # 2 - 1 subject - 4 runs outputs
        pipeline.subject_list = ['001']
        pipeline.run_list = ['01', '02', '03', '04']
        assert len(pipeline.get_preprocessing_outputs()) == 56
        assert len(pipeline.get_run_level_outputs()) == 0
        assert len(pipeline.get_subject_level_outputs()) == 7
        #assert len(pipeline.get_group_level_outputs()) == 63
        #assert len(pipeline.get_hypotheses_outputs()) == 18

        # 2 - 4 subjects outputs
        pipeline.subject_list = ['001', '002', '003', '004']
        pipeline.run_list = ['01', '02', '03', '04']
        assert len(pipeline.get_preprocessing_outputs()) == 224
        assert len(pipeline.get_run_level_outputs()) == 0
        assert len(pipeline.get_subject_level_outputs()) == 28
        #assert len(pipeline.get_group_level_outputs()) == 63
        #assert len(pipeline.get_hypotheses_outputs()) == 18

    @staticmethod
    @mark.pipeline_test
    def test_execution():
        """ Test the execution of a PipelineTeam3C6G and compare results """
        helpers.test_pipeline_evaluation('3C6G')
