#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_80GC' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_80GC.py
    pytest -q test_team_80GC.py -k <selected_test>
"""
from os.path import join, exists
from filecmp import cmp

from pytest import helpers, mark
from nipype import Workflow
from nipype.interfaces.base import Bunch

from narps_open.utils.configuration import Configuration
from narps_open.pipelines.team_80GC import PipelineTeam80GC

class TestPipelinesTeam80GC:
    """ A class that contains all the unit tests for the PipelineTeam80GC class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineTeam80GC object """

        pipeline = PipelineTeam80GC()

        # 1 - check the parameters
        assert pipeline.team_id == '80GC'

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
        """ Test the expected outputs of a PipelineTeam80GC object """
        pipeline = PipelineTeam80GC()
        # 1 - 1 subject outputs
        pipeline.subject_list = ['001']
        helpers.test_pipeline_outputs(pipeline, [0, 0, 2*1, 0, 18])

        # 2 - 4 subjects outputs
        pipeline.subject_list = ['001', '002', '003', '004']
        helpers.test_pipeline_outputs(pipeline, [0, 0, 2*4, 0, 18])

    @staticmethod
    @mark.unit_test
    def test_events_files(temporary_data_dir):
        """ Test the get_events_files method """

        # Get test files
        test_file = join(Configuration()['directories']['test_data'], 'pipelines', 'events.tsv')
        info = PipelineTeam80GC.get_events_files([test_file, test_file])

        # Compare bunches to expected
        bunch = info[0]
        assert isinstance(bunch, Bunch)
        assert bunch.conditions == ['gamble_run1']
        helpers.compare_float_2d_arrays(bunch.onsets, [[4.071, 11.834, 19.535, 27.535, 36.435]])
        helpers.compare_float_2d_arrays(bunch.durations, [[4.0, 4.0, 4.0, 4.0, 4.0]])
        assert bunch.amplitudes is None
        assert bunch.tmod is None
        assert bunch.pmod[0].name == ['gain_run1', 'loss_run1']
        assert bunch.pmod[0].poly == [1, 1]
        helpers.compare_float_2d_arrays(bunch.pmod[0].param,
            [[14.0, 34.0, 38.0, 10.0, 16.0], [6.0, 14.0, 19.0, 15.0, 17.0]])
        assert bunch.regressor_names is None
        assert bunch.regressors is None

        bunch = info[1]
        assert isinstance(bunch, Bunch)
        assert bunch.conditions == ['gamble_run2']
        helpers.compare_float_2d_arrays(bunch.onsets, [[4.071, 11.834, 19.535, 27.535, 36.435]])
        helpers.compare_float_2d_arrays(bunch.durations, [[4.0, 4.0, 4.0, 4.0, 4.0]])
        assert bunch.amplitudes is None
        assert bunch.tmod is None
        assert bunch.pmod[0].name == ['gain_run2', 'loss_run2']
        assert bunch.pmod[0].poly == [1, 1]
        helpers.compare_float_2d_arrays(bunch.pmod[0].param,
            [[14.0, 34.0, 38.0, 10.0, 16.0], [6.0, 14.0, 19.0, 15.0, 17.0]])
        assert bunch.regressor_names is None
        assert bunch.regressors is None

    @staticmethod
    @mark.unit_test
    def test_events_arguments():
        """ Test the get_events_arguments method """

    @staticmethod
    @mark.unit_test
    def test_confounds_file(temporary_data_dir):
        """ Test the get_confounds_file method """

        confounds_file = join(
            Configuration()['directories']['test_data'], 'pipelines', 'confounds.tsv')
        reference_file = join(
            Configuration()['directories']['test_data'], 'pipelines', 'team_80GC', 'confounds.tsv')

        # Create a Nipype Node using get_confounds_file
        test_get_confounds_file = Node(Function(
            function = PipelineTeam80GC.get_confounds_file,
            input_names = ['confounds_file', 'subject_id'],
            output_names = ['confounds_file']
            ), name = 'test_get_confounds_file')
        test_remove_dir_node.inputs.base_dir = temporary_data_dir
        test_remove_dir_node.inputs.confounds_file = confounds_file
        test_remove_dir_node.inputs.subject_id = 'sid'
        test_remove_dir_node.run()

        # Check confounds file was created
        created_confounds_file = join(
            temporary_data_dir, 'confounds_file_sub-sid.tsv')
        assert exists(created_confounds_file)

        # Check contents
        assert cmp(reference_file, created_confounds_file)

    @staticmethod
    @mark.unit_test
    def test_confounds_arguments():
        """ Test the get_confounds_arguments method """

    @staticmethod
    @mark.pipeline_test
    def test_execution():
        """ Test the execution of a PipelineTeam80GC and compare results """
        helpers.test_pipeline_evaluation('80GC')
