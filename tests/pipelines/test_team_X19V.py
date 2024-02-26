#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_X19V' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_X19V.py
    pytest -q test_team_X19V.py -k <selected_test>
"""
from os.path import join, exists
from filecmp import cmp

from pytest import helpers, mark
from nipype import Workflow
from nipype.interfaces.base import Bunch

from narps_open.utils.configuration import Configuration
from narps_open.pipelines.team_X19V import PipelineTeamX19V

TEMPORARY_DIR = join(Configuration()['directories']['test_runs'], 'test_X19V')

class TestPipelinesTeamX19V:
    """ A class that contains all the unit tests for the PipelineTeamX19V class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineTeamX19V object """

        pipeline = PipelineTeamX19V()

        # 1 - check the parameters
        assert pipeline.fwhm == 5.0
        assert pipeline.team_id == 'X19V'

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
        """ Test the expected outputs of a PipelineTeamX19V object """

        pipeline = PipelineTeamX19V()

        # 1 - 1 subject outputs
        pipeline.subject_list = ['001']
        helpers.test_pipeline_outputs(pipeline, [0, 4*1 + 4*4*4*1, 4*4*1 + 4*1, 8*4*2 + 4*4, 18])

        # 2 - 4 subjects outputs
        pipeline.subject_list = ['001', '002', '003', '004']
        helpers.test_pipeline_outputs(pipeline, [0, 4*4 + 4*4*4*4, 4*4*4 + 4*4, 8*4*2 + 4*4, 18])

    @staticmethod
    @mark.unit_test
    def test_subject_information():
        """ Test the get_subject_information method """

        # Get test files
        test_file = join(Configuration()['directories']['test_data'], 'pipelines', 'events.tsv')

        # Prepare several scenarii
        info_ok = PipelineTeamX19V.get_subject_information(test_file)

        # Compare bunches to expected
        bunch = info_ok[0]
        assert isinstance(bunch, Bunch)
        assert bunch.conditions == ['trial', 'gain', 'loss']
        helpers.compare_float_2d_arrays(bunch.onsets, [
            [4.071, 11.834, 19.535, 27.535, 36.435],
            [4.071, 11.834, 19.535, 27.535, 36.435],
            [4.071, 11.834, 19.535, 27.535, 36.435]])
        helpers.compare_float_2d_arrays(bunch.durations, [
            [4.0, 4.0, 4.0, 4.0, 4.0],
            [4.0, 4.0, 4.0, 4.0, 4.0],
            [4.0, 4.0, 4.0, 4.0, 4.0]])
        helpers.compare_float_2d_arrays(bunch.amplitudes, [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [-8.4, 11.6, 15.6, -12.4, -6.4],
            [-8.2, -0.2, 4.8, 0.8, 2.8]])
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
            Configuration()['directories']['test_data'], 'pipelines', 'team_X19V', 'confounds.tsv')

        # Get new confounds file
        PipelineTeamX19V.get_confounds_file(confounds_file, 'sid', 'rid', TEMPORARY_DIR)

        # Check confounds file was created
        created_confounds_file = join(
            TEMPORARY_DIR, 'confounds_files', 'confounds_file_sub-sid_run-rid.tsv')
        assert exists(created_confounds_file)

        # Check contents
        assert cmp(reference_file, created_confounds_file)

    @staticmethod
    @mark.pipeline_test
    def test_execution():
        """ Test the execution of a PipelineTeamX19V and compare results """
        helpers.test_pipeline_evaluation('X19V')
