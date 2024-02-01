#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_51PW' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_51PW.py
    pytest -q test_team_51PW.py -k <selected_test>
"""
from os.path import join, isfile

from pytest import helpers, mark
from numpy import isclose
from nipype import Workflow
from nipype.interfaces.base import Bunch

from narps_open.utils.configuration import Configuration
from narps_open.pipelines.team_51PW import PipelineTeam51PW

class TestPipelinesTeam51PW:
    """ A class that contains all the unit tests for the PipelineTeam51PW class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineTeam51PW object """

        pipeline = PipelineTeam51PW()

        # 1 - check the parameters
        assert pipeline.fwhm == 5.0
        assert pipeline.team_id == '51PW'
        assert pipeline.contrast_list == ['1', '2']
        assert pipeline.run_level_contasts == [
            ('effect_gain', 'T', ['gamble', 'gain', 'loss'], [0, 1, 0]),
            ('effect_loss', 'T', ['gamble', 'gain', 'loss'], [0, 0, 1])
        ]

        # 2 - check workflows
        assert isinstance(pipeline.get_preprocessing(), Workflow)
        assert isinstance(pipeline.get_run_level_analysis(), Workflow)
        assert isinstance(pipeline.get_subject_level_analysis(), Workflow)

        group_level = pipeline.get_group_level_analysis()
        assert len(group_level) == 3
        for sub_workflow in group_level:
            assert isinstance(sub_workflow, Workflow)

    @staticmethod
    @mark.unit_test
    def test_outputs():
        """ Test the expected outputs of a PipelineTeam51PW object """
        pipeline = PipelineTeam51PW()
        # 1 - 1 subject outputs
        pipeline.subject_list = ['001']
        helpers.test_pipeline_outputs(pipeline, [1*4, 4 + 4*2*4, 4*2, 0, 18])

        # 2 - 4 subjects outputs
        pipeline.subject_list = ['001', '002', '003', '004']
        helpers.test_pipeline_outputs(pipeline, [4*1*4, (4 + 4*2*4)*4, 4*2*4, 0, 18])

    @staticmethod
    @mark.unit_test
    def test_subject_information():
        """ Test the get_subject_information method """

        information = PipelineTeam51PW.get_subject_information(join(
            Configuration()['directories']['test_data'],
            'pipelines',
            'events.tsv'
            ))[0]

        assert isinstance(information, Bunch)
        assert information.conditions == ['gamble', 'gain', 'loss']

        reference_amplitudes = [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [14.0, 34.0, 38.0, 10.0, 16.0],
            [6.0, 14.0, 19.0, 15.0, 17.0]
            ]
        for reference_array, test_array in zip(reference_amplitudes, information.amplitudes):
            assert isclose(reference_array, test_array).all()

        reference_durations = [
            [4.0, 4.0, 4.0, 4.0, 4.0],
            [4.0, 4.0, 4.0, 4.0, 4.0],
            [4.0, 4.0, 4.0, 4.0, 4.0]
            ]
        for reference_array, test_array in zip(reference_durations, information.durations):
            assert isclose(reference_array, test_array).all()

        reference_onsets = [
            [4.071, 11.834, 19.535, 27.535, 36.435],
            [4.071, 11.834, 19.535, 27.535, 36.435],
            [4.071, 11.834, 19.535, 27.535, 36.435]
        ]
        for reference_array, test_array in zip(reference_onsets, information.onsets):
            assert isclose(reference_array, test_array).all()

    @staticmethod
    @mark.unit_test
    def test_confounds():
        """ Test the get_confounds method """

        # in_file, subject_id, run_id, working_dir
        out_filename = PipelineTeam51PW.get_confounds(join(
            Configuration()['directories']['test_data'], 'pipelines','confounds.tsv'),
            'subject_id',
            'run_id',
            Configuration()['directories']['test_runs']
            )

        assert isfile(out_filename)

        with open(out_filename, 'r', encoding = 'utf-8') as file:
            lines = file.readlines()
            assert len(lines) == 1
            assert lines[0] == '-2.56954e-05\t-0.00923735\t0.0549667\t0.000997278\t-0.00019745\t-0.000398988\t0.12437666391\t0.0462141151999999\t0.005774624\t-0.0439093598\t-0.075619539\t0.1754689153999999\n'

    @staticmethod
    @mark.unit_test
    def test_one_sample_t_test_regressors():
        """ Test the get_one_sample_t_test_regressors method """

        regressors = PipelineTeam51PW.get_one_sample_t_test_regressors(['001', '002'])
        assert regressors == {'group_mean': [1, 1]}

    @staticmethod
    @mark.unit_test
    def test_two_sample_t_test_regressors():
        """ Test the get_two_sample_t_test_regressors method """

        regressors, groups = PipelineTeam51PW.get_two_sample_t_test_regressors(
            ['001', '003'], # equalRange group
            ['002', '004'], # equalIndifference group
            ['001', '002', '003', '004'] # all subjects
            )
        assert regressors == dict(
                equalRange = [1, 0, 1, 0],
                equalIndifference = [0, 1, 0, 1]
            )
        assert groups == [1, 2, 1, 2]

    @staticmethod
    @mark.pipeline_test
    def test_execution():
        """ Test the execution of a PipelineTeam51PW and compare results """
        helpers.test_pipeline_evaluation('51PW')
