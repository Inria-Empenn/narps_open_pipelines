#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_R9K3' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_R9K3.py
    pytest -q test_team_R9K3.py -k <selected_test>
"""
from os.path import join, exists, abspath
from filecmp import cmp

from pytest import helpers, mark
from nipype import Workflow, Node
from nipype.interfaces.utility import Function
from nipype.interfaces.base import Bunch

from narps_open.utils.configuration import Configuration
from narps_open.pipelines.team_R9K3 import PipelineTeamR9K3

class TestPipelinesTeamR9K3:
    """ A class that contains all the unit tests for the PipelineTeamR9K3 class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineTeamR9K3 object """

        pipeline = PipelineTeamR9K3()

        # 1 - check the parameters
        assert pipeline.fwhm == 6.0
        assert pipeline.team_id == 'R9K3'

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
        """ Test the expected outputs of a PipelineTeamR9K3 object """
        pipeline = PipelineTeamR9K3()
        # 1 - 1 subject outputs
        pipeline.subject_list = ['001']
        helpers.test_pipeline_outputs(pipeline, [4, 0, 5, 42, 18])

        # 2 - 4 subjects outputs
        pipeline.subject_list = ['001', '002', '003', '004']
        helpers.test_pipeline_outputs(pipeline, [16, 0, 20, 42, 18])

    @staticmethod
    @mark.unit_test
    def test_confounds_file(temporary_data_dir):
        """ Test the get_confounds_file method """

        # Test files
        in_confounds_file = abspath(join(Configuration()['directories']['test_data'],
            'pipelines', 'confounds.tsv'))

        # Reference file
        ref_file = abspath(join(Configuration()['directories']['test_data'],
            'pipelines', 'team_R9K3', 'confounds.tsv'))

        # Create average values file
        confounds_node = Node(Function(
            input_names = ['confounds_file', 'subject_id', 'run_id'],
            output_names = ['out_file'],
            function = PipelineTeamR9K3.get_confounds_file),
            name = 'confounds_node')
        confounds_node.base_dir = temporary_data_dir
        confounds_node.inputs.confounds_file = in_confounds_file
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
        information = PipelineTeamR9K3.get_subject_information(test_event_file)

        assert isinstance(information, Bunch)
        assert information.conditions == ['trial']

        helpers.compare_float_2d_arrays([[0.0, 0.0, 0.0, 0.0, 0.0]], information.durations)
        helpers.compare_float_2d_arrays(
            [[4.071, 11.834, 19.535, 27.535, 36.435]],
            information.onsets)
        paramateric_modulation = information.pmod[0]
        assert isinstance(paramateric_modulation, Bunch)
        assert paramateric_modulation.name == ['gain', 'loss']
        assert paramateric_modulation.poly == [1, 1]
        helpers.compare_float_2d_arrays(
            [[0.368421053, 0.894736842, 1.0, 0.263157895, 0.421052632],
            [0.315789474, 0.736842105, 1.0, 0.789473684, 0.894736842]],
            paramateric_modulation.param)

    @staticmethod
    @mark.pipeline_test
    def test_execution():
        """ Test the execution of a PipelineTeamR9K3 and compare results """
        helpers.test_pipeline_evaluation('R9K3')
