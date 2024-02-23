#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_UK24' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_UK24.py
    pytest -q test_team_UK24.py -k <selected_test>
"""

from os.path import join, exists, abspath
from filecmp import cmp

from pytest import helpers, mark
from nipype import Workflow, Node
from nipype.interfaces.utility import Function
from nipype.interfaces.base import Bunch

from narps_open.utils.configuration import Configuration
from narps_open.pipelines.team_UK24 import PipelineTeamUK24

class TestPipelinesTeamUK24:
    """ A class that contains all the unit tests for the PipelineTeamUK24 class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineTeamUK24 object """

        pipeline = PipelineTeamUK24()

        # 1 - check the parameters
        assert pipeline.fwhm == 4.0
        assert pipeline.team_id == 'UK24'

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
        """ Test the expected outputs of a PipelineTeamUK24 object """
        pipeline = PipelineTeamUK24()
        # 1 - 1 subject outputs
        pipeline.subject_list = ['001']
        helpers.test_pipeline_outputs(pipeline, [21, 0, 5, 42, 18])

        # 2 - 4 subjects outputs
        pipeline.subject_list = ['001', '002', '003', '004']
        helpers.test_pipeline_outputs(pipeline, [84, 0, 20, 42, 18])

    @staticmethod
    @mark.unit_test
    def test_average_values(temporary_data_dir):
        """ Test the get_average_values method
            For this test, we created the two following files by downsampling output files
                from preprocessing nodes :
                - func_resampled-32.nii (smoothed functional files)
                - mask_resampled-32.nii (white matter mask)
            Voxel dimension was multiplied by 32, number of slices was reduced to 4.
        """
        # Test files
        test_in_file = abspath(join(Configuration()['directories']['test_data'],
            'pipelines', 'team_UK24', 'func_resampled-32.nii'))
        test_mask_file = abspath(join(Configuration()['directories']['test_data'],
            'pipelines', 'team_UK24', 'mask_resampled-32.nii'))

        # Reference file
        ref_file = abspath(join(Configuration()['directories']['test_data'],
            'pipelines', 'team_UK24', 'reference_average_values.txt'))

        # Create average values file
        average_node = Node(Function(
            input_names = ['in_file', 'mask', 'subject_id', 'run_id', 'out_file_suffix'],
            output_names = ['out_file'],
            function = PipelineTeamUK24.get_average_values),
            name = 'average_node')
        average_node.base_dir = temporary_data_dir
        average_node.inputs.in_file = test_in_file
        average_node.inputs.mask = test_mask_file
        average_node.inputs.subject_id = 'sid'
        average_node.inputs.run_id = 'rid'
        average_node.inputs.out_file_suffix = 'suffix.txt'
        average_node.run()

        # Check file was created
        created_average_values_file = abspath(join(
            temporary_data_dir, average_node.name, 'sub-sid_run-rid_suffix.txt'))
        assert exists(created_average_values_file)

        # Check contents
        assert cmp(ref_file, created_average_values_file)

    @staticmethod
    @mark.unit_test
    def test_confounds_file(temporary_data_dir):
        """ Test the get_confounds_file method """

        # Test files
        test_average_values_csf = abspath(join(Configuration()['directories']['test_data'],
            'pipelines', 'team_UK24', 'average_values_csf.txt'))
        test_average_values_wm = abspath(join(Configuration()['directories']['test_data'],
            'pipelines', 'team_UK24', 'average_values_wm.txt'))
        test_framewise_displacement = abspath(join(Configuration()['directories']['test_data'],
            'pipelines', 'team_UK24', 'framewise_displacement.txt'))
        test_realignment_parameters = abspath(join(Configuration()['directories']['test_data'],
            'pipelines', 'team_UK24', 'realignment_parameters.txt'))

        # Reference file
        ref_file = abspath(join(Configuration()['directories']['test_data'],
            'pipelines', 'team_UK24', 'confounds.tsv'))

        # Create average values file
        confounds_node = Node(Function(
            input_names = [
                'framewise_displacement_file', 'wm_average_file', 'csf_average_file',
                'realignement_parameters', 'subject_id', 'run_id'
                ],
            output_names = ['out_file'],
            function = PipelineTeamUK24.get_confounds_file),
            name = 'confounds_node')
        confounds_node.base_dir = temporary_data_dir
        confounds_node.inputs.framewise_displacement_file = test_framewise_displacement
        confounds_node.inputs.wm_average_file = test_average_values_wm
        confounds_node.inputs.csf_average_file = test_average_values_csf
        confounds_node.inputs.realignement_parameters = test_realignment_parameters
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
        information = PipelineTeamUK24.get_subject_information(test_event_file)

        assert isinstance(information, Bunch)
        assert information.conditions == ['gain', 'loss', 'no_gain_no_loss']

        helpers.compare_float_2d_arrays([
            [4.0, 4.0],
            [4.0, 4.0],
            [4.0, 4.0, 4.0]
            ],
            information.durations)
        helpers.compare_float_2d_arrays([
            [4.071, 11.834],
            [4.071, 11.834],
            [19.535, 27.535, 36.435]
            ],
            information.onsets)

        paramateric_modulation = information.pmod[0]
        assert isinstance(paramateric_modulation, Bunch)
        assert paramateric_modulation.name == ['gain_value', 'gain_rt']
        assert paramateric_modulation.poly == [1, 1]
        helpers.compare_float_2d_arrays([
            [14.0, 34.0],
            [2.388, 2.289]
            ],
            paramateric_modulation.param)

        paramateric_modulation = information.pmod[1]
        assert isinstance(paramateric_modulation, Bunch)
        assert paramateric_modulation.name == ['loss_value', 'loss_rt']
        assert paramateric_modulation.poly == [1, 1]
        helpers.compare_float_2d_arrays([
            [6.0, 14.0],
            [2.388, 2.289]
            ],
            paramateric_modulation.param)

        paramateric_modulation = information.pmod[2]
        assert isinstance(paramateric_modulation, Bunch)
        assert paramateric_modulation.name == ['no_gain_no_loss_rt']
        assert paramateric_modulation.poly == [1]
        helpers.compare_float_2d_arrays([
            [0.0, 2.08, 2.288]
            ],
            paramateric_modulation.param)

    @staticmethod
    @mark.pipeline_test
    def test_execution():
        """ Test the execution of a PipelineTeamUK24 and compare results """
        helpers.test_pipeline_evaluation('UK24')
