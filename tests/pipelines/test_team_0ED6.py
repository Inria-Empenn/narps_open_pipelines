#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_0ED6' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_0ED6.py
    pytest -q test_team_0ED6.py -k <selected_test>
"""

from os.path import join, exists, abspath
from filecmp import cmp

from pytest import helpers, mark
from nipype import Workflow, Node
from nipype.interfaces.utility import Function
from nipype.interfaces.base import Bunch

from narps_open.utils.configuration import Configuration
from narps_open.pipelines.team_0ED6 import PipelineTeam0ED6

class TestPipelinesTeam0ED6:
    """ A class that contains all the unit tests for the PipelineTeam0ED6 class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineTeam0ED6 object """

        pipeline = PipelineTeam0ED6()

        # 1 - check the parameters
        assert pipeline.fwhm == 5.0
        assert pipeline.team_id == '0ED6'

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
        """ Test the expected outputs of a PipelineTeam0ED6 object """
        pipeline = PipelineTeam0ED6()
        # 1 - 1 subject outputs
        pipeline.subject_list = ['001']
        helpers.test_pipeline_outputs(pipeline, [20, 0, 9, 84, 18])

        # 2 - 4 subjects outputs
        pipeline.subject_list = ['001', '002', '003', '004']
        helpers.test_pipeline_outputs(pipeline, [80, 0, 36, 84, 18])

    @staticmethod
    @mark.unit_test
    def test_confounds_file(temporary_data_dir):
        """ Test the get_confounds_file method """

        # Test files
        test_dvars = abspath(join(Configuration()['directories']['test_data'],
            'pipelines', 'team_0ED6', 'swrusub-001_task-MGT_run-01_bold_dvars_std.tsv'))
        test_realignment_parameters = abspath(join(Configuration()['directories']['test_data'],
            'pipelines', 'team_0ED6', 'rp_sub-001_task-MGT_run-01_bold.txt'))

        # Reference file
        ref_file = abspath(join(Configuration()['directories']['test_data'],
            'pipelines', 'team_0ED6', 'confounds_file_sub-sid_run-rid.tsv'))

        # Create average values file
        confounds_node = Node(Function(
            input_names = ['dvars_file', 'realignement_parameters', 'subject_id', 'run_id'],
            output_names = ['out_file'],
            function = PipelineTeam0ED6.get_confounds_file),
            name = 'confounds_node')
        confounds_node.base_dir = temporary_data_dir
        confounds_node.inputs.realignement_parameters = test_realignment_parameters
        confounds_node.inputs.dvars_file = test_dvars
        confounds_node.inputs.subject_id = 'sid'
        confounds_node.inputs.run_id = 'rid'
        confounds_node.run()

        # Check file was created
        created_confounds_file = abspath(join(
            temporary_data_dir, confounds_node.name, 'confounds_file_sub-sid_run-rid.tsv'))
        assert exists(created_confounds_file)

        # Check contents
        assert cmp(ref_file, created_confounds_file)

    @staticmethod
    @mark.unit_test
    def test_subject_information():
        """ Test the get_subject_information method """

        # Test with 'gain'
        test_event_file = join(
            Configuration()['directories']['test_data'], 'pipelines', 'events.tsv')
        information = PipelineTeam0ED6.get_subject_information(test_event_file)

        assert isinstance(information, Bunch)
        assert information.conditions == ['task']

        helpers.compare_float_2d_arrays([
            [4.0, 4.0, 4.0, 4.0, 4.0]
            ],
            information.durations)
        helpers.compare_float_2d_arrays([
            [4.071, 11.834, 19.535, 27.535, 36.435]
            ],
            information.onsets)

        paramateric_modulation = information.pmod[0]
        assert isinstance(paramateric_modulation, Bunch)
        assert paramateric_modulation.name == ['gain', 'loss', 'reaction_time']
        assert paramateric_modulation.poly == [1, 1, 1]
        helpers.compare_float_2d_arrays([
            [14.0, 34.0, 38.0, 10.0, 16.0],
            [6.0, 14.0, 19.0, 15.0, 17.0],
            [2.388, 2.289, 0.0, 2.08, 2.288]
            ],
            paramateric_modulation.param)

    @staticmethod
    @mark.pipeline_test
    def test_execution():
        """ Test the execution of a PipelineTeam0ED6 and compare results """
        helpers.test_pipeline_evaluation('0ED6')
