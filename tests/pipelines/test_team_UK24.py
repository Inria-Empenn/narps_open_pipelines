#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_UK24' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_UK24.py
    pytest -q test_team_UK24.py -k <selected_test>
"""

from os.path import join

from pytest import helpers, mark
from numpy import isclose
from nipype import Workflow
from nipype.interfaces.base import Bunch

from narps_open.utils.configuration import Configuration
from narps_open.pipelines.team_UK24 import PipelineTeamUK24

class TestPipelinesTeamUK24:
    """ A class that contains all the unit tests for the PipelineTeamUK24 class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineTeamUK24 object """

        pipeline = PipelineTeamUK24()

        # 1 - check the parameters
        assert pipeline.fwhm == 4.0
        assert pipeline.team_id == 'UK24'

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
        """ Test the expected outputs of a PipelineTeamUK24 object """
        pipeline = PipelineTeamUK24()
        # 1 - 1 subject outputs
        pipeline.subject_list = ['001']
        helpers.test_pipeline_outputs(pipeline, [21, 0, 8, 53, 18])

        # 2 - 4 subjects outputs
        pipeline.subject_list = ['001', '002', '003', '004']
        helpers.test_pipeline_outputs(pipeline, [84, 0, 32, 53, 18])

    @staticmethod
    @mark.unit_test
    def test_average_values():
        """ Test the get_average_values method """

        # Test files
        test_wm_file = join(Configuration()['directories']['test_data'],
            'pipelines', 'team_UK24', '.tsv')
        test_csf_file = join(Configuration()['directories']['test_data'],
            'pipelines', 'team_UK24', '.tsv')
        test_rp_file = join(Configuration()['directories']['test_data'],
            'pipelines', 'team_UK24', '.tsv')
        test_fd_file = join(Configuration()['directories']['test_data'],
            'pipelines', 'team_UK24', '.tsv')

        # Reference file
        ref_file = join(Configuration()['directories']['test_data'],
            'pipelines', 'team_UK24', '.tsv')

        # Exec function
        PipelineTeamUK24.get_subject_information(

            )

    @staticmethod
    @mark.unit_test
    def test_confounds_file:
        """ Test the get_confounds_file method """

    @staticmethod
    @mark.unit_test
    def test_subject_information():
        """ Test the get_subject_information method """

        # Test with 'gain'
        test_event_file = join(
            Configuration()['directories']['test_data'], 'pipelines', 'events.tsv')
        information = PipelineTeamUK24.get_subject_information(
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
        test_event_file = join(
            Configuration()['directories']['test_data'], 'pipelines', 'events.tsv')
        information = PipelineTeamUK24.get_subject_information(
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
        """ Test the execution of a PipelineTeamUK24 and compare results """
        helpers.test_pipeline_evaluation('UK24')
