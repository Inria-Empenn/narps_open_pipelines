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

from narps_open.pipelines.team_1KB2 import PipelineTeam1KB2

class TestPipelinesTeam1KB2:
    """ A class that contains all the unit tests for the PipelineTeam2T6S class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineTeam1KB2 object """

        pipeline = PipelineTeam1KB2()

        # 1 - check the parameters
        assert pipeline.team_id == '1KB2'

        # 2 - check workflows
        assert isinstance(pipeline.get_preprocessing(), Workflow)
        assert isinstance(pipeline.get_run_level_analysis(), Workflow)
        assert isinstance(pipeline.registration(), Workflow)
        assert isinstance(pipeline.get_subject_level_analysis(), Workflow)
        group_level = pipeline.get_group_level_analysis()

        assert len(group_level) == 3
        for sub_workflow in group_level:
            assert isinstance(sub_workflow, Workflow)

    @staticmethod
    @mark.unit_test
    def test_outputs():
        """ Test the expected outputs of a PipelineTeam2T6S object """
        pipeline = PipelineTeam1KB2()
        # 1 - 1 suject outputs, 1 run
        pipeline.subject_list = ['001']
        pipeline.run_list = ['01']
        pipeline.contrast_list = ['1','2']
        assert len(pipeline.get_preprocessing_outputs()) == 6
        #assert len(pipeline.get_run_level_outputs()) == 0
        #assert len(pipeline.get_registration()) == 0
        #assert len(pipeline.get_subject_level_outputs()) == 
        #assert len(pipeline.get_group_level_outputs()) == 84

        # 2 - 1 suject outputs, 4 runs
        pipeline.subject_list = ['001']
        pipeline.run_list = ['01', '02', '03', '04']
        pipeline.contrast_list = ['1','2']
        assert len(pipeline.get_preprocessing_outputs()) == 24
        #assert len(pipeline.get_run_level_outputs()) == 0
        #assert len(pipeline.get_registration()) == 0
        #assert len(pipeline.get_subject_level_outputs()) == 36
        #assert len(pipeline.get_group_level_outputs()) == 84

        # 3 - 4 suject outputs, 4 runs
        pipeline.subject_list = ['001', '002', '003', '004']
        pipeline.run_list = ['01', '02', '03', '04']
        pipeline.contrast_list = ['1','2']
        assert len(pipeline.get_preprocessing_outputs()) == 96
        #assert len(pipeline.get_run_level_outputs()) == 0
        #assert len(pipeline.get_registration()) == 0
        #assert len(pipeline.get_subject_level_outputs()) == 36
        #assert len(pipeline.get_group_level_outputs()) == 84

    @staticmethod
    @mark.pipeline_test
    def test_execution():
        """ Test the execution of a PipelineTeam1KB2 and compare results """
        results_4_subjects = helpers.test_pipeline(
            '1KB2',
            '/references/',
            '/data/',
            '/output/',
            4)
        assert mean(results_4_subjects) > .003
