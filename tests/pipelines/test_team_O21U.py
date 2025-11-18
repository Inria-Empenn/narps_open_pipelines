#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_O21U' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_O21U.py
    pytest -q test_team_O21U.py -k <selected_test>
"""
from os import mkdir
from os.path import join, exists, abspath
from filecmp import cmp

from pytest import helpers, mark
from nipype import Workflow, Node, Function
from nipype.interfaces.base import Bunch

from narps_open.utils.configuration import Configuration
from narps_open.pipelines.team_O21U import PipelineTeamO21U

class TestPipelinesTeamO21U:
    """ A class that contains all the unit tests for the PipelineTeamO21U class."""

    @staticmethod
    @mark.unit_test
    def test_create(temporary_data_dir):
        """ Test the creation of a PipelineTeamO21U object """

        pipeline = PipelineTeamO21U()

        # Workaround to have existing base_directory for datagrabber nodes
        pipeline.directories.output_dir = temporary_data_dir
        pipeline.directories.dataset_dir = temporary_data_dir
        mkdir(join(temporary_data_dir, 'run_level_analysis'))
        mkdir(join(temporary_data_dir, 'subject_level_analysis'))
        mkdir(join(temporary_data_dir, 'derivatives'))
        mkdir(join(temporary_data_dir, 'derivatives', 'fmriprep'))

        # 1 - check the parameters
        assert pipeline.fwhm == 5.0
        assert pipeline.team_id == 'O21U'

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
        """ Test the expected outputs of a PipelineTeamO21U object """

        pipeline = PipelineTeamO21U()

        # 1 - 1 subject outputs
        pipeline.subject_list = ['001']
        helpers.test_pipeline_outputs(pipeline, [4, 4*1*2*4, 4*2*1 + 1, 6*2*2 + 3*2, 18])

        # 2 - 4 subjects outputs
        pipeline.subject_list = ['001', '002', '003', '004']
        helpers.test_pipeline_outputs(pipeline, [4*4, 4*4*2*4, 4*2*4 + 1, 6*2*2 + 3*2, 18])

    @staticmethod
    @mark.unit_test
    def test_subject_information():
        """ Test the get_subject_information method """

        # Get test files
        test_file = join(Configuration()['directories']['test_data'], 'pipelines', 'events.tsv')

        # Prepare several scenarii
        info_missed = PipelineTeamO21U.get_subject_information(test_file)

        # Compare bunches to expected
        bunch = info_missed[0]
        assert isinstance(bunch, Bunch)
        assert bunch.conditions == ['trial', 'gain', 'loss']
        helpers.compare_float_2d_arrays(bunch.onsets, [
            [4.071, 11.834, 19.535, 27.535, 36.435],
            [4.071, 11.834, 19.535, 27.535, 36.435],
            [4.071, 11.834, 19.535, 27.535, 36.435]
            ])
        helpers.compare_float_2d_arrays(bunch.durations, [
            [4.0, 4.0, 4.0, 4.0, 4.0],
            [4.0, 4.0, 4.0, 4.0, 4.0],
            [4.0, 4.0, 4.0, 4.0, 4.0]
            ])
        helpers.compare_float_2d_arrays(bunch.amplitudes, [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [14.0, 34.0, 38.0, 10.0, 16.0],
            [6.0, 14.0, 19.0, 15.0, 17.0]
            ])

    @staticmethod
    @mark.unit_test
    def test_confounds_file(temporary_data_dir):
        """ Test the get_confounds_file method """

        # Get input and reference output file
        confounds_file = abspath(join(
            Configuration()['directories']['test_data'], 'pipelines', 'confounds.tsv'))
        reference_file = abspath(join(
            Configuration()['directories']['test_data'],
            'pipelines', 'team_O21U', 'confounds.tsv'))

        # Create new confounds file
        confounds_node = Node(Function(
            input_names = ['in_file'],
            output_names = ['confounds_file'],
            function = PipelineTeamO21U.get_confounds_file),
            name = 'confounds_node')
        confounds_node.base_dir = temporary_data_dir
        confounds_node.inputs.in_file = confounds_file
        confounds_node.run()

        # Check confounds file was created
        created_confounds_file = abspath(join(
            temporary_data_dir, confounds_node.name, 'confounds_file.tsv'))
        assert exists(created_confounds_file)

        # Check contents
        assert cmp(reference_file, created_confounds_file)

    @staticmethod
    @mark.unit_test
    def test_get_subject_level_contrasts():
        """ Test the get_subject_level_contrasts method """

        subjects = ['s1', 's2']
        runs = ['r1', 'r2', 'r3']
        contrasts, regressors = PipelineTeamO21U.get_subject_level_contrasts(subjects, runs)
        assert contrasts == [
            ['', 'T', ['ev1', 'ev2'], [1.0, 0.0]],
            ['', 'T', ['ev1', 'ev2'], [0.0, 1.0]]
            ]
        assert regressors == {
            'ev1': [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            'ev2': [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        } 

    @staticmethod
    @mark.unit_test
    def test_get_one_sample_t_test_regressors():
        """ Test the get_one_sample_t_test_regressors method """

        subjects = ['s1', 's2']
        assert PipelineTeamO21U.get_one_sample_t_test_regressors(subjects) == {
            'group_mean': [1, 1]
            }
        subjects = ['s1', 's2', 's3', 's4']
        assert PipelineTeamO21U.get_one_sample_t_test_regressors(subjects) == {
            'group_mean': [1, 1, 1, 1]
            }

    @staticmethod
    @mark.unit_test
    def test_get_two_sample_t_test_regressors():
        """ Test the get_two_sample_t_test_regressors method """

        subjects = ['s1', 's2']
        er_group = ['s1']
        ei_group = ['s2']
        regressors, groups = PipelineTeamO21U.get_two_sample_t_test_regressors(
            subjects, er_group, ei_group)
        assert regressors == {
            'equalRange': [1, 0],
            'equalIndifference': [0, 1]
            }
        assert groups == [1, 2]

        subjects = ['s1', 's2', 's3', 's4']
        er_group = ['s1', 's3']
        ei_group = ['s2', 's4']
        regressors, groups = PipelineTeamO21U.get_two_sample_t_test_regressors(
            subjects, er_group, ei_group)
        assert regressors == {
            'equalRange': [1, 0, 1, 0],
            'equalIndifference': [0, 1, 0, 1]
            }
        assert groups == [1, 2, 1, 2]

    @staticmethod
    @mark.pipeline_test
    def test_execution():
        """ Test the execution of a PipelineTeamO21U and compare results """
        helpers.test_pipeline_evaluation('O21U')
