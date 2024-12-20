#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_B23O' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_B23O.py
    pytest -q test_team_B23O.py -k <selected_test>
"""
from os.path import join, exists, abspath
from filecmp import cmp

from pytest import helpers, mark
from nipype import Workflow, Node, Function
from nipype.interfaces.base import Bunch

from narps_open.utils.configuration import Configuration
from narps_open.pipelines.team_B23O import PipelineTeamB23O

class TestPipelinesTeamB23O:
    """ A class that contains all the unit tests for the PipelineTeamB23O class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineTeamB23O object """

        pipeline = PipelineTeamB23O()

        # 1 - check the parameters
        assert pipeline.team_id == 'B23O'

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
        """ Test the expected outputs of a PipelineTeamB23O object """

        pipeline = PipelineTeamB23O()

        # 1 - 1 subject outputs
        pipeline.subject_list = ['001']
        helpers.test_pipeline_outputs(pipeline, [0, 4*2*4*1, 4*2*1, 6*2*2 + 3*2, 18])

        # 2 - 4 subjects outputs
        pipeline.subject_list = ['001', '002', '003', '004']
        helpers.test_pipeline_outputs(pipeline, [0, 4*2*4*4, 4*2*4, 6*2*2 + 3*2, 18])

    @staticmethod
    @mark.unit_test
    def test_subject_information():
        """ Test the get_subject_information method """

        # Get test files
        test_file = join(Configuration()['directories']['test_data'], 'pipelines', 'events.tsv')

        # Prepare several scenarii
        info = PipelineTeamB23O.get_subject_information(test_file)

        # Compare bunches to expected
        bunch = info[0]
        assert isinstance(bunch, Bunch)
        assert bunch.conditions == ['trial', 'gain', 'loss']
        helpers.compare_float_2d_arrays(bunch.onsets, [
            [4.071, 11.834, 19.535, 27.535, 36.435],
            [4.071, 11.834, 19.535, 27.535, 36.435],
            [4.071, 11.834, 19.535, 27.535, 36.435],
            ])
        helpers.compare_float_2d_arrays(bunch.durations, [
            [4.0, 4.0, 4.0, 4.0, 4.0],
            [4.0, 4.0, 4.0, 4.0, 4.0],
            [4.0, 4.0, 4.0, 4.0, 4.0],
            ])
        helpers.compare_float_2d_arrays(bunch.amplitudes, [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [14.0, 34.0, 38.0, 10.0, 16.0],
            [6.0, 14.0, 19.0, 15.0, 17.0],
            ])

    @staticmethod
    @mark.unit_test
    def test_confounds_file(temporary_data_dir):
        """ Test the get_confounds_file method """

        # Get input and reference output file
        confounds_file = join(
            Configuration()['directories']['test_data'], 'pipelines', 'confounds.tsv')
        reference_file = join(
            Configuration()['directories']['test_data'],
            'pipelines', 'team_B23O', 'confounds.tsv')

        # Create new confounds file
        confounds_node = Node(Function(
            input_names = ['filepath', 'subject_id', 'run_id'],
            output_names = ['confounds_file'],
            function = PipelineTeamB23O.get_confounds_file),
            name = 'confounds_node')
        confounds_node.base_dir = temporary_data_dir
        confounds_node.inputs.filepath = confounds_file
        confounds_node.inputs.subject_id = 'sid'
        confounds_node.inputs.run_id = 'rid'
        confounds_node.run()

        # Check confounds file was created
        created_confounds_file = abspath(join(
            temporary_data_dir, confounds_node.name, 'confounds_file_sub-sid_run-rid.tsv'))
        assert exists(created_confounds_file)

        # Check contents
        assert cmp(reference_file, created_confounds_file)

    @staticmethod
    @mark.pipeline_test
    def test_execution():
        """ Test the execution of a PipelineTeamB23O and compare results """
        helpers.test_pipeline_evaluation('B23O')
