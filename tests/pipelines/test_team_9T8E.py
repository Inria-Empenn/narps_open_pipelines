#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_9T8E' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_9T8E.py
    pytest -q test_team_9T8E.py -k <selected_test>
"""
from os.path import join, exists
from filecmp import cmp

from pytest import helpers, mark
from nipype import Workflow
from nipype.interfaces.base import Bunch

from narps_open.utils.configuration import Configuration
from narps_open.pipelines.team_9T8E import PipelineTeam9T8E

class TestPipelinesTeam9T8E:
    """ A class that contains all the unit tests for the PipelineTeam9T8E class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineTeam9T8E object """

        pipeline = PipelineTeam9T8E()

        # 1 - check the parameters
        assert pipeline.fwhm == 8.0
        assert pipeline.team_id == '9T8E'

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
        """ Test the expected outputs of a PipelineTeam9T8E object """
        pipeline = PipelineTeam9T8E()
        # 1 - 1 subject outputs
        pipeline.subject_list = ['001']
        helpers.test_pipeline_outputs(pipeline, [0, 0, 5, 6*2*2 + 4*2, 18])

        # 2 - 4 subjects outputs
        pipeline.subject_list = ['001', '002', '003', '004']
        helpers.test_pipeline_outputs(pipeline, [0, 0, 20, 6*2*2 + 4*2, 18])

    @staticmethod
    @mark.unit_test
    def test_subject_information():
        """ Test the get_subject_information method """

        # Get test files
        test_file = join(Configuration()['directories']['test_data'], 'pipelines', 'events.tsv')
        test_file_resp = join(
            Configuration()['directories']['test_data'], 'pipelines', 'events_resp.tsv')
        info = PipelineTeam9T8E.get_subject_information([test_file, test_file_resp])

        # Compare bunches to expected
        bunch = info[0]
        assert isinstance(bunch, Bunch)
        assert bunch.conditions == ['accept_run1', 'reject_run1', 'noresp_run1']
        helpers.compare_float_2d_arrays(bunch.onsets, [
            [4.071, 11.834],
            [27.535, 36.435],
            [19.535]
            ])
        helpers.compare_float_2d_arrays(bunch.durations, [
            [4.0, 4.0],
            [4.0, 4.0],
            [4.0]
            ])
        assert bunch.amplitudes is None
        assert bunch.tmod is None
        assert bunch.pmod[0].name == ['gain', 'loss', 'reaction_time']
        assert bunch.pmod[0].poly == [1, 1, 1]
        helpers.compare_float_2d_arrays(bunch.pmod[0].param, [
            [14.0, 34.0],
            [6.0, 14.0],
            [2.388, 2.289]
            ])
        assert bunch.pmod[1].name == ['gain', 'loss', 'reaction_time']
        assert bunch.pmod[1].poly == [1, 1, 1]
        helpers.compare_float_2d_arrays(bunch.pmod[1].param, [
            [10.0, 16.0],
            [15.0, 17.0],
            [2.08, 2.288]
            ])
        assert bunch.regressor_names is None
        assert bunch.regressors is None

        bunch = info[1]
        assert isinstance(bunch, Bunch)
        assert bunch.conditions == ['accept_run2', 'reject_run2']
        helpers.compare_float_2d_arrays(bunch.onsets, [
            [4.071, 11.834],
            [27.535, 36.435]
            ])
        helpers.compare_float_2d_arrays(bunch.durations, [
            [4.0, 4.0],
            [4.0, 4.0]
            ])
        assert bunch.amplitudes is None
        assert bunch.tmod is None
        assert bunch.pmod[0].name == ['gain', 'loss', 'reaction_time']
        assert bunch.pmod[0].poly == [1, 1, 1]
        helpers.compare_float_2d_arrays(bunch.pmod[0].param, [
            [14.0, 34.0],
            [6.0, 14.0],
            [2.388, 2.289]
            ])
        assert bunch.pmod[1].name == ['gain', 'loss', 'reaction_time']
        assert bunch.pmod[1].poly == [1, 1, 1]
        helpers.compare_float_2d_arrays(bunch.pmod[1].param, [
            [10.0, 16.0],
            [15.0, 17.0],
            [2.08, 2.288]
            ])
        assert bunch.regressor_names is None
        assert bunch.regressors is None

    @staticmethod
    @mark.unit_test
    def test_confounds_file(temporary_data_dir):
        """ Test the get_confounds_file method """

        confounds_file = join(
            Configuration()['directories']['test_data'], 'pipelines', 'confounds.tsv')
        reference_file = join(
            Configuration()['directories']['test_data'], 'pipelines', 'team_9T8E', 'confounds.tsv')

        # Get new confounds file
        PipelineTeam9T8E.get_confounds_file(confounds_file, 'sid', 'rid', temporary_data_dir)

        # Check confounds file was created
        created_confounds_file = join(
            temporary_data_dir, 'confounds_files', 'confounds_file_sub-sid_run-rid.tsv')
        assert exists(created_confounds_file)

        # Check contents
        assert cmp(reference_file, created_confounds_file)

    @staticmethod
    @mark.pipeline_test
    def test_execution():
        """ Test the execution of a PipelineTeam9T8E and compare results """
        helpers.test_pipeline_evaluation('9T8E')
