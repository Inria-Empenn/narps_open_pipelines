#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_98BT' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_98BT.py
    pytest -q test_team_98BT.py -k <selected_test>
"""
from os.path import join, exists
from filecmp import cmp

from pytest import helpers, mark
from nipype import Workflow
from nipype.interfaces.base import Bunch

from narps_open.utils.configuration import Configuration
from narps_open.pipelines.team_98BT import PipelineTeam98BT

class TestPipelinesTeam98BT:
    """ A class that contains all the unit tests for the PipelineTeam98BT class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineTeam98BT object """

        pipeline = PipelineTeam98BT()

        # 1 - check the parameters
        assert pipeline.fwhm == 8.0
        assert pipeline.team_id == '98BT'

        # 2 - check workflows
        processing = pipeline.get_preprocessing()
        assert len(processing) == 2
        for sub_workflow in processing:
            assert isinstance(sub_workflow, Workflow)

        assert pipeline.get_run_level_analysis() is None
        assert isinstance(pipeline.get_subject_level_analysis(), Workflow)

        group_level = pipeline.get_group_level_analysis()
        assert len(group_level) == 3
        for sub_workflow in group_level:
            assert isinstance(sub_workflow, Workflow)

    @staticmethod
    @mark.unit_test
    def test_outputs():
        """ Test the expected outputs of a PipelineTeam98BT object """
        pipeline = PipelineTeam98BT()
        # 1 - 1 subject outputs
        pipeline.subject_list = ['001']
        helpers.test_pipeline_outputs(pipeline, [1 + 1*1 + 4*1*4,0,9,84,18])

        # 2 - 4 subjects outputs
        pipeline.subject_list = ['001', '002', '003', '004']
        helpers.test_pipeline_outputs(pipeline, [1 + 4*1 + 4*4*4,0,36,84,18])

    @staticmethod
    @mark.unit_test
    def test_fieldmap_info():
        """ Test the get_fieldmap_info method """

        filedmap_file_1 = join(
            Configuration()['directories']['test_data'], 'pipelines', 'phasediff_1.json')
        filedmap_file_2 = join(
            Configuration()['directories']['test_data'], 'pipelines', 'phasediff_2.json')

        test_result = PipelineTeam98BT.get_fieldmap_info(
            filedmap_file_1, ['magnitude_1', 'magnitude_2'])
        assert test_result[0] == (0.00492, 0.00738)
        assert test_result[1] == 'magnitude_1'
        test_result = PipelineTeam98BT.get_fieldmap_info(
            filedmap_file_2, ['magnitude_1', 'magnitude_2'])
        assert test_result[0] == (0.00492, 0.00738)
        assert test_result[1] == 'magnitude_2'

    @staticmethod
    @mark.unit_test
    def test_parameters_files(temporary_data_dir):
        """ Test the get_parameters_files method
            For this test, we created the two following files by downsampling output files
                from the preprocessing pipeline :
                - wc2sub-001_T1w-32.nii (white matter file)
                - uasub-001_task-MGT_run-01_bold_resampled-32.nii (motion corrected file)
            Voxel dimension was multiplied by 32, number of slices was reduced to 4.
        """
        parameters_file = join(
            Configuration()['directories']['test_data'], 'pipelines', 'confounds.tsv')
        func_file = join(Configuration()['directories']['test_data'], 'pipelines',
            'team_98BT', 'uasub-001_task-MGT_run-01_bold_resampled-32.nii')
        wc2_file = join(Configuration()['directories']['test_data'], 'pipelines',
            'team_98BT', 'wc2sub-001_T1w-32.nii')
        reference_file = join(
            Configuration()['directories']['test_data'], 'pipelines',
            'team_98BT', 'parameters_file.tsv')

        # Get new parameters file
        PipelineTeam98BT.get_parameters_file(
            parameters_file, wc2_file, func_file, 'sid', 'rid', temporary_data_dir)

        # Check parameters file was created
        created_parameters_file = join(
            temporary_data_dir, 'parameters_files', 'parameters_file_sub-sid_run-rid.tsv')
        assert exists(created_parameters_file)

        # Check contents
        assert cmp(reference_file, created_parameters_file)

    @staticmethod
    @mark.unit_test
    def test_subject_information():
        """ Test the get_subject_information method """

        # Get test files
        test_file = join(Configuration()['directories']['test_data'], 'pipelines', 'events.tsv')

        bunch = PipelineTeam98BT.get_subject_information(test_file, 1)

        # Compare bunches to expected
        assert isinstance(bunch, Bunch)
        assert bunch.conditions == ['gamble_run1']
        helpers.compare_float_2d_arrays(bunch.onsets, [
            [4.071, 11.834, 19.535, 27.535, 36.435]])
        helpers.compare_float_2d_arrays(bunch.durations, [
            [4.0, 4.0, 4.0, 4.0, 4.0]])
        assert bunch.amplitudes is None
        assert bunch.tmod is None
        assert bunch.regressor_names is None
        assert bunch.regressors is None
        pmod = bunch.pmod[0]
        assert isinstance(pmod, Bunch)
        assert pmod.name == ['gain', 'loss', 'answers']
        assert pmod.poly == [1, 1, 1]
        helpers.compare_float_2d_arrays(pmod.param, [
            [14.0, 34.0, 38.0, 10.0, 16.0],
            [6.0, 14.0, 19.0, 15.0, 17.0],
            [1, 2, -2, -2, -1]
            ])

    @staticmethod
    @mark.pipeline_test
    def test_execution():
        """ Test the execution of a PipelineTeam98BT and compare results """
        helpers.test_pipeline_evaluation('98BT')
