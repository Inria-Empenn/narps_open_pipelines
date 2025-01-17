#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_0H5E' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_0H5E.py
    pytest -q test_team_0H5E.py -k <selected_test>
"""
from os.path import join

from pytest import helpers, mark
from nipype import Workflow
from nipype.interfaces.base import Bunch

from narps_open.utils.configuration import Configuration
from narps_open.pipelines.team_0H5E import PipelineTeam0H5E

class TestPipelinesTeam0H5E:
    """ A class that contains all the unit tests for the PipelineTeam0H5E class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineTeam0H5E object """

        pipeline = PipelineTeam0H5E()

        # 1 - check the parameters
        assert pipeline.fwhm == 9.0
        assert pipeline.team_id == '0H5E'

        # 2 - check workflows
        assert isinstance(pipeline.get_preprocessing(), Workflow)
        assert pipeline.get_run_level_analysis() is None

        subject_level = pipeline.get_subject_level_analysis()
        assert len(subject_level) == 2
        for sub_workflow in subject_level:
            assert isinstance(sub_workflow, Workflow)

        group_level = pipeline.get_group_level_analysis()
        assert len(group_level) == 3
        for sub_workflow in group_level:
            assert isinstance(sub_workflow, Workflow)

    @staticmethod
    @mark.unit_test
    def test_outputs():
        """ Test the expected outputs of a PipelineTeam0H5E object """
        pipeline = PipelineTeam0H5E()
        # 1 - 1 subject outputs
        pipeline.subject_list = ['001']
        helpers.test_pipeline_outputs(pipeline, [1*4*2, 0, 2 + 2 + 2, 2*2*8 + 2*5, 18])

        # 2 - 4 subjects outputs
        pipeline.subject_list = ['001', '002', '003', '004']
        helpers.test_pipeline_outputs(pipeline, [4*4*2, 0, 4 * (2 + 2 + 2), 2*2*8 + 2*5, 18])

    @staticmethod
    @mark.unit_test
    def test_subject_information():
        """ Test the get_subject_information method """

        # Get test files
        test_file = join(Configuration()['directories']['test_data'], 'pipelines', 'events.tsv')

        # Create a bunch
        bunch = PipelineTeam0H5E.get_subject_information(test_file, 1, 'gain')

        # Compare bunch to expected
        assert isinstance(bunch, Bunch)
        assert bunch.conditions == ['trial_run1']
        helpers.compare_float_2d_arrays(bunch.onsets, [
            [0.071, 7.834, 15.535, 23.535, 32.435]])
        helpers.compare_float_2d_arrays(bunch.durations, [
            [4.0, 4.0, 4.0, 4.0, 4.0]])
        assert bunch.amplitudes is None
        assert bunch.tmod is None
        assert bunch.regressor_names is None
        assert bunch.regressors is None
        pmod = bunch.pmod[0]
        assert isinstance(pmod, Bunch)
        assert pmod.name == ['gain', 'loss']
        assert pmod.poly == [1, 1]
        helpers.compare_float_2d_arrays(pmod.param, [
            [14.0, 34.0, 38.0, 10.0, 16.0],
            [6.0, 14.0, 19.0, 15.0, 17.0]
            ])

        # New bunch
        bunch = PipelineTeam0H5E.get_subject_information(test_file, 3, 'loss')

        # Compare bunch to expected
        assert isinstance(bunch, Bunch)
        assert bunch.conditions == ['trial_run3']
        helpers.compare_float_2d_arrays(bunch.onsets, [
            [0.071, 7.834, 15.535, 23.535, 32.435]])
        helpers.compare_float_2d_arrays(bunch.durations, [
            [4.0, 4.0, 4.0, 4.0, 4.0]])
        assert bunch.amplitudes is None
        assert bunch.tmod is None
        assert bunch.regressor_names is None
        assert bunch.regressors is None
        pmod = bunch.pmod[0]
        assert isinstance(pmod, Bunch)
        assert pmod.name == ['loss', 'gain']
        assert pmod.poly == [1, 1]
        helpers.compare_float_2d_arrays(pmod.param, [
            [6.0, 14.0, 19.0, 15.0, 17.0],
            [14.0, 34.0, 38.0, 10.0, 16.0]
            ])

    @staticmethod
    @mark.pipeline_test
    def test_execution():
        """ Test the execution of a PipelineTeam0H5E and compare results """
        helpers.test_pipeline_evaluation('0H5E')
