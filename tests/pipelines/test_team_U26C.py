#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_U26C' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_U26C.py
    pytest -q test_team_U26C.py -k <selected_test>
"""
from os import mkdir
from os.path import join, exists
from shutil import rmtree
from filecmp import cmp

from pytest import helpers, mark, fixture
from numpy import isclose
from nipype import Workflow
from nipype.interfaces.base import Bunch

from narps_open.utils.configuration import Configuration
from narps_open.pipelines.team_U26C import PipelineTeamU26C

TEMPORARY_DIR = join(Configuration()['directories']['test_runs'], 'test_U26C')

@fixture
def remove_test_dir():
    """ A fixture to remove temporary directory created by tests """

    rmtree(TEMPORARY_DIR, ignore_errors = True)
    mkdir(TEMPORARY_DIR)
    yield # test runs here
    #rmtree(TEMPORARY_DIR, ignore_errors = True)

def compare_float_2d_arrays(array_1, array_2):
    """ Assert array_1 and array_2 are close enough """

    assert len(array_1) == len(array_2)
    for reference_array, test_array in zip(array_1, array_2):
        assert len(reference_array) == len(test_array)
        assert isclose(reference_array, test_array).all()

class TestPipelinesTeamU26C:
    """ A class that contains all the unit tests for the PipelineTeamU26C class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineTeamU26C object """

        pipeline = PipelineTeamU26C()

        # 1 - check the parameters
        assert pipeline.fwhm == 5.0
        assert pipeline.team_id == 'U26C'

        # 2 - check workflows
        assert pipeline.get_preprocessing() is None
        assert pipeline.get_run_level_analysis() is None
        assert isinstance(pipeline.get_subject_level_analysis(), Workflow)
        group_level = pipeline.get_group_level_analysis()

        """assert len(group_level) == 3
        for sub_workflow in group_level:
            assert isinstance(sub_workflow, Workflow)"""

    @staticmethod
    @mark.unit_test
    def test_outputs():
        """ Test the expected outputs of a PipelineTeamU26C object """
        pipeline = PipelineTeamU26C()
        # 1 - 1 subject outputs
        """pipeline.subject_list = ['001']
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
        assert len(pipeline.get_hypotheses_outputs()) == 18"""

    @staticmethod
    @mark.unit_test
    def test_subject_information():
        """ Test the get_subject_information method """

        # Get test files
        test_file = join(Configuration()['directories']['test_data'], 'pipelines', 'events.tsv')
        info = PipelineTeamU26C.get_subject_information([test_file, test_file])

        # Compare bunches to expected
        bunch = info[0]
        assert isinstance(bunch, Bunch)
        assert bunch.conditions == ['gamble_run1']
        compare_float_2d_arrays(bunch.onsets, [[4.071, 11.834, 19.535, 27.535, 36.435]])
        compare_float_2d_arrays(bunch.durations, [[4.0, 4.0, 4.0, 4.0, 4.0]])
        assert bunch.amplitudes == None
        assert bunch.tmod == None
        assert bunch.pmod[0].name == ['gain_run1', 'loss_run1']
        assert bunch.pmod[0].poly == [1, 1]
        compare_float_2d_arrays(bunch.pmod[0].param, [[14.0, 34.0, 38.0, 10.0, 16.0], [6.0, 14.0, 19.0, 15.0, 17.0]])
        assert bunch.regressor_names == None
        assert bunch.regressors == None

        bunch = info[1]
        assert isinstance(bunch, Bunch)
        assert bunch.conditions == ['gamble_run2']
        compare_float_2d_arrays(bunch.onsets, [[4.071, 11.834, 19.535, 27.535, 36.435]])
        compare_float_2d_arrays(bunch.durations, [[4.0, 4.0, 4.0, 4.0, 4.0]])
        assert bunch.amplitudes == None
        assert bunch.tmod == None
        assert bunch.pmod[0].name == ['gain_run2', 'loss_run2']
        assert bunch.pmod[0].poly == [1, 1]
        compare_float_2d_arrays(bunch.pmod[0].param, [[14.0, 34.0, 38.0, 10.0, 16.0], [6.0, 14.0, 19.0, 15.0, 17.0]])
        assert bunch.regressor_names == None
        assert bunch.regressors == None

    @staticmethod
    @mark.unit_test
    def test_confounds_file(remove_test_dir):
        """ Test the get_confounds_file method """

        confounds_file = join(
            Configuration()['directories']['test_data'], 'pipelines', 'confounds.tsv')
        reference_file = join(
            Configuration()['directories']['test_data'], 'pipelines', 'team_U26C', 'confounds.tsv')

        # Get new confounds file
        PipelineTeamU26C.get_confounds_file(confounds_file, 'sid', 'rid', TEMPORARY_DIR)

        # Check confounds file was created
        created_confounds_file = join(
            TEMPORARY_DIR, 'confounds_files', 'confounds_file_sub-sid_run-rid.tsv')
        assert exists(created_confounds_file)

        # Check contents
        assert cmp(reference_file, created_confounds_file)

    @staticmethod
    @mark.pipeline_test
    def test_execution():
        """ Test the execution of a PipelineTeamU26C and compare results """
        helpers.test_pipeline_evaluation('U26C')
