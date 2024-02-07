#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_J7F9' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_J7F9.py
    pytest -q test_team_J7F9.py -k <selected_test>
"""
from os.path import join, exists
from filecmp import cmp

from pytest import helpers, mark
from nipype import Workflow
from nipype.interfaces.base import Bunch

from narps_open.utils.configuration import Configuration
from narps_open.pipelines.team_J7F9 import PipelineTeamJ7F9

TEMPORARY_DIR = join(Configuration()['directories']['test_runs'], 'test_J7F9')

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
        helpers.test_pipeline_outputs(pipeline, [0, 0, 7, 63, 18])

        # 2 - 4 subjects outputs
        pipeline.subject_list = ['001', '002', '003', '004']
        helpers.test_pipeline_outputs(pipeline, [0, 0, 28, 63, 18])

    @staticmethod
    @mark.unit_test
    def test_subject_information():
        """ Test the get_subject_information method """

        # Get test files
        test_file = join(Configuration()['directories']['test_data'], 'pipelines', 'events.tsv')
        test_file_2 = join(Configuration()['directories']['test_data'],
            'pipelines', 'team_J7F9', 'events_resp.tsv')

        # Prepare several scenarii
        info_missed = PipelineTeamJ7F9.get_subject_information([test_file, test_file])
        info_ok = PipelineTeamJ7F9.get_subject_information([test_file_2, test_file_2])
        info_half = PipelineTeamJ7F9.get_subject_information([test_file_2, test_file])

        # Compare bunches to expected
        bunch = info_missed[0]
        assert isinstance(bunch, Bunch)
        assert bunch.conditions == ['trial', 'missed']
        helpers.compare_float_2d_arrays(bunch.onsets, [
            [4.071, 11.834, 19.535, 27.535, 36.435], [19.535]])
        helpers.compare_float_2d_arrays(bunch.durations, [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0]])
        assert bunch.amplitudes is None
        assert bunch.tmod is None
        assert bunch.pmod[0].name == ['gain', 'loss']
        assert bunch.pmod[0].poly == [1, 1]
        helpers.compare_float_2d_arrays(bunch.pmod[0].param, [
            [-8.4, 11.6, 15.6, -12.4, -6.4], [-8.2, -0.2, 4.8, 0.8, 2.8]])
        assert bunch.regressor_names is None
        assert bunch.regressors is None

        bunch = info_missed[1]
        assert isinstance(bunch, Bunch)
        assert bunch.conditions == ['trial', 'missed']
        helpers.compare_float_2d_arrays(bunch.onsets, [
            [4.071, 11.834, 19.535, 27.535, 36.435], [19.535]])
        helpers.compare_float_2d_arrays(bunch.durations, [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0]])
        assert bunch.amplitudes is None
        assert bunch.tmod is None
        assert bunch.pmod[0].name == ['gain', 'loss']
        assert bunch.pmod[0].poly == [1, 1]
        helpers.compare_float_2d_arrays(bunch.pmod[0].param, [
            [-8.4, 11.6, 15.6, -12.4, -6.4], [-8.2, -0.2, 4.8, 0.8, 2.8]])
        assert bunch.regressor_names is None
        assert bunch.regressors is None

        bunch = info_ok[0]
        assert isinstance(bunch, Bunch)
        assert bunch.conditions == ['trial']
        helpers.compare_float_2d_arrays(bunch.onsets, [[4.071, 11.834, 27.535, 36.435]])
        helpers.compare_float_2d_arrays(bunch.durations, [[0.0, 0.0, 0.0, 0.0]])
        assert bunch.amplitudes is None
        assert bunch.tmod is None
        assert bunch.pmod[0].name == ['gain', 'loss']
        assert bunch.pmod[0].poly == [1, 1]
        helpers.compare_float_2d_arrays(bunch.pmod[0].param, [
            [-4.5, 15.5, -8.5, -2.5], [-7.0, 1.0, 2.0, 4.0]])
        assert bunch.regressor_names is None
        assert bunch.regressors is None

        bunch = info_ok[1]
        assert isinstance(bunch, Bunch)
        assert bunch.conditions == ['trial']
        helpers.compare_float_2d_arrays(bunch.onsets, [[4.071, 11.834, 27.535, 36.435]])
        helpers.compare_float_2d_arrays(bunch.durations, [[0.0, 0.0, 0.0, 0.0]])
        assert bunch.amplitudes is None
        assert bunch.tmod is None
        assert bunch.pmod[0].name == ['gain', 'loss']
        assert bunch.pmod[0].poly == [1, 1]
        helpers.compare_float_2d_arrays(bunch.pmod[0].param, [
            [-4.5, 15.5, -8.5, -2.5], [-7.0, 1.0, 2.0, 4.0]])
        assert bunch.regressor_names is None
        assert bunch.regressors is None

        bunch = info_half[0]
        assert isinstance(bunch, Bunch)
        assert bunch.conditions == ['trial', 'missed']
        helpers.compare_float_2d_arrays(bunch.onsets, [[4.071, 11.834, 27.535, 36.435], []])
        helpers.compare_float_2d_arrays(bunch.durations, [[0.0, 0.0, 0.0, 0.0], []])
        assert bunch.amplitudes is None
        assert bunch.tmod is None
        assert bunch.pmod[0].name == ['gain', 'loss']
        assert bunch.pmod[0].poly == [1, 1]
        helpers.compare_float_2d_arrays(
            bunch.pmod[0].param,
            [[-6.666666666666668, 13.333333333333332, -10.666666666666668, -4.666666666666668],
             [-7.666666666666666, 0.3333333333333339, 1.333333333333334, 3.333333333333334]])
        assert bunch.regressor_names is None
        assert bunch.regressors is None

        bunch = info_half[1]
        assert isinstance(bunch, Bunch)
        assert bunch.conditions == ['trial', 'missed']
        helpers.compare_float_2d_arrays(bunch.onsets, [
            [4.071, 11.834, 19.535, 27.535, 36.435], [19.535]])
        helpers.compare_float_2d_arrays(bunch.durations, [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0]])
        assert bunch.amplitudes is None
        assert bunch.tmod is None
        assert bunch.pmod[0].name == ['gain', 'loss']
        assert bunch.pmod[0].poly == [1, 1]
        helpers.compare_float_2d_arrays(
            bunch.pmod[0].param,
            [[
            -6.666666666666668,
            13.333333333333332,
            17.333333333333332,
            -10.666666666666668,
            -4.666666666666668],
            [-7.666666666666666,
            0.3333333333333339,
            5.333333333333334,
            1.333333333333334,
            3.333333333333334]])
        assert bunch.regressor_names is None
        assert bunch.regressors is None

    @staticmethod
    @mark.unit_test
    @mark.parametrize('remove_test_dir', TEMPORARY_DIR)
    def test_confounds_file(remove_test_dir):
        """ Test the get_confounds_file method """

        confounds_file = join(
            Configuration()['directories']['test_data'], 'pipelines', 'confounds.tsv')
        reference_file = join(
            Configuration()['directories']['test_data'], 'pipelines', 'team_J7F9', 'confounds.tsv')

        # Get new confounds file
        PipelineTeamJ7F9.get_confounds_file(confounds_file, 'sid', 'rid', TEMPORARY_DIR)

        # Check confounds file was created
        created_confounds_file = join(
            TEMPORARY_DIR, 'confounds_files', 'confounds_file_sub-sid_run-rid.tsv')
        assert exists(created_confounds_file)

        # Check contents
        assert cmp(reference_file, created_confounds_file)

    @staticmethod
    @mark.pipeline_test
    def test_execution():
        """ Test the execution of a PipelineTeamJ7F9 and compare results """
        helpers.test_pipeline_evaluation('J7F9')
