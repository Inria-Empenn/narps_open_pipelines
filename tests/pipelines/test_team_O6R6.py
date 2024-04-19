#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_O6R6' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_O6R6.py
    pytest -q test_team_O6R6.py -k <selected_test>
"""
from os.path import join, exists, abspath
from filecmp import cmp

from pytest import helpers, mark
from nipype import Workflow, Node, Function
from nipype.interfaces.base import Bunch

from narps_open.utils.configuration import Configuration
from narps_open.pipelines.team_O6R6 import PipelineTeamO6R6

class TestPipelinesTeamO6R6:
    """ A class that contains all the unit tests for the PipelineTeamO6R6 class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineTeamO6R6 object """

        pipeline = PipelineTeamO6R6()

        # 1 - check the parameters
        assert pipeline.team_id == 'O6R6'

        # 2 - check workflows
        assert pipeline.get_preprocessing() is None
        assert isinstance(pipeline.get_run_level_analysis(), Workflow)
        assert isinstance(pipeline.get_subject_level_analysis(), Workflow)
        group_level = pipeline.get_group_level_analysis()
        assert len(group_level) == 3
        for sub_workflow in group_level:
            assert isinstance(sub_workflow, Workflow)

    @staticmethod
    @mark.unit_test
    def test_outputs():
        """ Test the expected outputs of a PipelineTeamO6R6 object """

        pipeline = PipelineTeamO6R6()

        # 1 - 1 subject outputs
        pipeline.subject_list = ['001']
        helpers.test_pipeline_outputs(pipeline, [0, 2*1*4*4, 2*4*1 + 2*1, 8*4*2 + 4*4, 18])

        # 2 - 4 subjects outputs
        pipeline.subject_list = ['001', '002', '003', '004']
        helpers.test_pipeline_outputs(pipeline, [0, 2*4*4*4, 2*4*4 + 2*4, 8*4*2 + 4*4, 18])

    @staticmethod
    @mark.unit_test
    def test_subject_information():
        """ Test the get_subject_information method """

        # Get test files
        test_file = join(Configuration()['directories']['test_data'], 'pipelines', 'events.tsv')

        # Prepare several scenarii
        info_ei = PipelineTeamO6R6.get_subject_information(test_file, 'equalIndifference')
        info_er = PipelineTeamO6R6.get_subject_information(test_file, 'equalRange')

        # Compare bunches to expected
        bunch = info_ei[0]
        assert isinstance(bunch, Bunch)
        assert bunch.conditions == ['trial', 'gain_trial', 'loss_trial']
        helpers.compare_float_2d_arrays(bunch.onsets, [
            [4.071, 11.834, 27.535, 36.435],
            [4.071, 11.834],
            [27.535, 36.435]
            ])        
        helpers.compare_float_2d_arrays(bunch.durations, [
            [2.388, 2.289, 2.08, 2.288],
            [2.388, 2.289],
            [2.08, 2.288]
            ])
        helpers.compare_float_2d_arrays(bunch.amplitudes, [
            [1.0, 1.0, 1.0, 1.0],
            [3.0, 13.0],
            [3.5, 4.5]
            ])

        # Compare bunches to expected
        bunch = info_er[0]
        assert isinstance(bunch, Bunch)
        assert bunch.conditions == ['trial', 'gain_trial', 'loss_trial']
        helpers.compare_float_2d_arrays(bunch.onsets, [
            [4.071, 11.834, 27.535, 36.435],
            [4.071, 11.834],
            [27.535, 36.435]
            ])
        helpers.compare_float_2d_arrays(bunch.durations, [
            [2.388, 2.289, 2.08, 2.288],
            [2.388, 2.289],
            [2.08, 2.288]
            ])
        helpers.compare_float_2d_arrays(bunch.amplitudes, [
            [1.0, 1.0, 1.0, 1.0],
            [10.0, 30.0],
            [11.0, 13.0]
            ])

    @staticmethod
    @mark.pipeline_test
    def test_execution():
        """ Test the execution of a PipelineTeamO6R6 and compare results """
        helpers.test_pipeline_evaluation('O6R6')
