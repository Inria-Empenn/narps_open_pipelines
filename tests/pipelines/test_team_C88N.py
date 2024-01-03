#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_C88N' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_C88N.py
    pytest -q test_team_C88N.py -k <selected_test>
"""

from os.path import join

from pytest import helpers, mark
from numpy import isclose
from nipype import Workflow
from nipype.interfaces.base import Bunch

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

        assert len(group_level) == 5
        for sub_workflow in group_level:
            assert isinstance(sub_workflow, Workflow)

    @staticmethod
    @mark.unit_test
    def test_outputs():
        """ Test the expected outputs of a PipelineTeamC88N object """
        pipeline = PipelineTeamC88N()
        # 1 - 1 subject outputs
        pipeline.subject_list = ['001']
        assert len(pipeline.get_preprocessing_outputs()) == 0
        assert len(pipeline.get_run_level_outputs()) == 0
        assert len(pipeline.get_subject_level_outputs()) == 8
        assert len(pipeline.get_group_level_outputs()) == 53
        assert len(pipeline.get_hypotheses_outputs()) == 18

        # 2 - 4 subjects outputs
        pipeline.subject_list = ['001', '002', '003', '004']
        assert len(pipeline.get_preprocessing_outputs()) == 0
        assert len(pipeline.get_run_level_outputs()) == 0
        assert len(pipeline.get_subject_level_outputs()) == 32
        assert len(pipeline.get_group_level_outputs()) == 53
        assert len(pipeline.get_hypotheses_outputs()) == 18

    @staticmethod
    @mark.unit_test
    def test_subject_information():
        """ Test the get_subject_information method """

        # Test with 'gain'
        test_event_file = join(Configuration()['directories']['test_data'], 'pipelines', 'events.tsv')
        information = PipelineTeamC88N.get_subject_information(
            [test_event_file, test_event_file],
            'gain'
            )[0]

        assert isinstance(information, Bunch)
        assert information.conditions == ['trial']

        reference_durations = [[0.0, 0.0, 0.0, 0.0]]
        assert len(reference_durations) == len(information.durations)
        for reference_array, test_array in zip(reference_durations, information.durations):
            assert isclose(reference_array, test_array).all()

        reference_onsets = [[4.071, 11.834, 27.535, 36.435]]
        assert len(reference_onsets) == len(information.onsets)
        for reference_array, test_array in zip(reference_onsets, information.onsets):
            assert isclose(reference_array, test_array).all()

        paramateric_modulation = information.pmod[0]

        assert isinstance(paramateric_modulation, Bunch)
        assert paramateric_modulation.name == ['loss', 'gain']
        assert paramateric_modulation.poly == [1, 1]

        reference_param = [[6.0, 14.0, 15.0, 17.0], [14.0, 34.0, 10.0, 16.0]]
        assert len(reference_param) == len(paramateric_modulation.param)
        for reference_array, test_array in zip(reference_param, paramateric_modulation.param):
            assert isclose(reference_array, test_array).all()

        # Test with 'loss'
        test_event_file = join(Configuration()['directories']['test_data'], 'pipelines', 'events.tsv')
        information = PipelineTeamC88N.get_subject_information(
            [test_event_file, test_event_file],
            'loss'
            )[0]

        assert isinstance(information, Bunch)
        assert information.conditions == ['trial']

        reference_durations = [[0.0, 0.0, 0.0, 0.0]]
        assert len(reference_durations) == len(information.durations)
        for reference_array, test_array in zip(reference_durations, information.durations):
            assert isclose(reference_array, test_array).all()

        reference_onsets = [[4.071, 11.834, 27.535, 36.435]]
        assert len(reference_onsets) == len(information.onsets)
        for reference_array, test_array in zip(reference_onsets, information.onsets):
            assert isclose(reference_array, test_array).all()

        paramateric_modulation = information.pmod[0]

        assert isinstance(paramateric_modulation, Bunch)
        assert paramateric_modulation.name == ['gain', 'loss']
        assert paramateric_modulation.poly == [1, 1]

        reference_param = [[14.0, 34.0, 10.0, 16.0], [6.0, 14.0, 15.0, 17.0]]
        assert len(reference_param) == len(paramateric_modulation.param)
        for reference_array, test_array in zip(reference_param, paramateric_modulation.param):
            assert isclose(reference_array, test_array).all()

    @staticmethod
    @mark.pipeline_test
    def test_execution():
        """ Test the execution of a PipelineTeamC88N and compare results """
        helpers.test_pipeline_evaluation('C88N')
