#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_J7F9' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_J7F9.py
    pytest -q test_team_J7F9.py -k <selected_test>
"""
from os.path import join

from pytest import helpers, mark
from numpy import isclose
from nipype import Workflow
from nipype.interfaces.base import Bunch

from narps_open.utils.configuration import Configuration
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
    @mark.unit_test
    def test_subject_information():
        """ Test the get_subject_information method """

        test_event_file = join(Configuration()['directories']['test_data'], 'pipelines', 'events.tsv')
        information = PipelineTeamJ7F9.get_subject_information([test_event_file, test_event_file])

        for run_id in [0, 1]:
            bunch = information [run_id] 

            assert isinstance(bunch, Bunch)
            assert bunch.conditions == ['trial', 'missed']

            reference_durations = [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]
                ]
            assert len(reference_durations) == len(bunch.durations)
            for reference_array, test_array in zip(reference_durations, bunch.durations):
                assert isclose(reference_array, test_array).all()

            reference_onsets = [
                [4.071, 11.834, 19.535, 27.535, 36.435],
                [19.535]
                ]
            assert len(reference_onsets) == len(bunch.onsets)
            for reference_array, test_array in zip(reference_onsets, bunch.onsets):
                assert isclose(reference_array, test_array).all()

            paramateric_modulation = bunch.pmod[0]

            assert isinstance(paramateric_modulation, Bunch)
            assert paramateric_modulation.name == ['gain', 'loss']
            assert paramateric_modulation.poly == [1, 1]

            reference_param = [
                [-8.4, 11.6, 15.6, -12.4, -6.4],
                [-8.2, -0.2, 4.8, 0.8, 2.8]
                ]
            assert len(reference_param) == len(paramateric_modulation.param)
            for reference_array, test_array in zip(reference_param, paramateric_modulation.param):
                assert isclose(reference_array, test_array).all()

    @staticmethod
    @mark.pipeline_test
    def test_execution():
        """ Test the execution of a PipelineTeamJ7F9 and compare results """
        helpers.test_pipeline_evaluation('J7F9')
