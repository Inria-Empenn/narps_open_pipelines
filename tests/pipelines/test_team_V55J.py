#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_V55J' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_V55J.py
    pytest -q test_team_V55J.py -k <selected_test>
"""
from os import remove
from os.path import join, exists
from filecmp import cmp

from pytest import helpers, mark
from nipype import Workflow
from nipype.interfaces.base import Bunch

from narps_open.utils.configuration import Configuration
from narps_open.pipelines.team_V55J import PipelineTeamV55J

class TestPipelinesTeamV55J:
    """ A class that contains all the unit tests for the PipelineTeamV55J class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineTeamV55J object """

        pipeline = PipelineTeamV55J()

        # 1 - check the parameters
        assert pipeline.fwhm == 6.0
        assert pipeline.team_id == 'V55J'

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
        """ Test the expected outputs of a PipelineTeamV55J object """
        pipeline = PipelineTeamV55J()
        # 1 - 1 subject outputs
        pipeline.subject_list = ['001']
        helpers.test_pipeline_outputs(pipeline, [7*4, 0, 5, 8*2*2 + 5*2, 18])

        # 2 - 4 subjects outputs
        pipeline.subject_list = ['001', '002', '003', '004']
        helpers.test_pipeline_outputs(pipeline, [7*4*4, 0, 20, 8*2*2 + 5*2, 18])

    @staticmethod
    @mark.unit_test
    def test_union_mask(temporary_data_dir):
        """ Test the union_mask method
            For this test, we created the three following files by using a small region of interest
            from the wc* preprocessiong output files :
                - wc1sub-020_T1w_roi.nii.gz (grey-matter mask)
                - wc2sub-020_T1w_roi.nii.gz (white matter mask)
                - wc3sub-020_T1w_roi.nii.gz (csf mask)
            A region of interest of 10x10x10 voxels was selected from the original files, using:
            > fslroi wc1sub-020_T1w.nii wc1sub-020_T1w_roi.nii 80 10 78 10 48 10
            > fslroi wc2sub-020_T1w.nii wc2sub-020_T1w_roi.nii 80 10 78 10 48 10
            > fslroi wc3sub-020_T1w.nii wc3sub-020_T1w_roi.nii 80 10 78 10 48 10
        """
        # Test files
        wc1_file = join(Configuration()['directories']['test_data'], 'pipelines',
            'team_V55J', 'wc1sub-020_T1w_roi.nii.gz')
        wc2_file = join(Configuration()['directories']['test_data'], 'pipelines',
            'team_V55J', 'wc2sub-020_T1w_roi.nii.gz')
        wc3_file = join(Configuration()['directories']['test_data'], 'pipelines',
            'team_V55J', 'wc3sub-020_T1w_roi.nii.gz')

        # Expected results
        reference_file_1 = join(
            Configuration()['directories']['test_data'], 'pipelines',
            'team_V55J', 'mask_1.nii')
        reference_file_2 = join(
            Configuration()['directories']['test_data'], 'pipelines',
            'team_V55J', 'mask_2.nii')

        # Create binarized union mask file
        union_mask = Node(Function(
            input_names = ['masks', 'threshold'],
            output_names = ['mask'],
            function = PipelineTeamUK24.get_average_values),
            name = 'union_mask')
        union_mask.base_dir = temporary_data_dir
        union_mask.inputs.masks = [wc1_file, wc2_file, wc3_file]
        union_mask.inputs.threshold = 0.3
        union_mask.run()

        # Check mask file was created & its contents
        created_mask = abspath(join(
            temporary_data_dir, union_mask.name, 'mask.nii'))
        assert exists(created_mask)
        assert cmp(reference_file_1, created_mask)
        # Remove file
        remove(created_mask)

        # New test
        union_mask.inputs.masks = [wc3_file, wc1_file, wc2_file]
        union_mask.inputs.threshold = 0.5
        union_mask.run()

        # Check mask file was created & its contents
        created_mask = abspath(join(
            temporary_data_dir, union_mask.name, 'mask.nii'))
        assert exists(created_mask)
        assert cmp(reference_file_2, created_mask)

    @staticmethod
    @mark.unit_test
    def test_subject_information():
        """ Test the get_subject_information method """

        # Get test files
        test_file = join(Configuration()['directories']['test_data'], 'pipelines', 'events.tsv')

        bunch = PipelineTeamV55J.get_subject_information(test_file)

        # Compare bunches to expected
        assert isinstance(bunch, Bunch)
        assert bunch.conditions == ['trial', 'accepting', 'rejecting']
        helpers.compare_float_2d_arrays(bunch.onsets, [
            [4.071, 11.834, 19.535, 27.535, 36.435],
            [6.459, 14.123],
            [29.615, 38.723]
            ])
        helpers.compare_float_2d_arrays(bunch.durations, [
            [4.0, 4.0, 4.0, 4.0, 4.0],
            [0.0, 0.0],
            [0.0, 0.0]
            ])
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
        assert bunch.pmod[1] is None
        assert bunch.pmod[2] is None

    @staticmethod
    @mark.pipeline_test
    def test_execution():
        """ Test the execution of a PipelineTeamV55J and compare results """
        helpers.test_pipeline_evaluation('V55J')
