#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_98BT' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_98BT.py
    pytest -q test_team_98BT.py -k <selected_test>
"""

from pytest import helpers, mark
from nipype import Workflow

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
        helpers.test_pipeline_outputs(pipeline, [0,0,9,84,18])

        # 2 - 4 subjects outputs
        pipeline.subject_list = ['001', '002', '003', '004']
        helpers.test_pipeline_outputs(pipeline, [0,0,36,84,18])

    @staticmethod
    @mark.pipeline_test
    def test_fieldmap_info():
        """ Test the get_fieldmap_info method """

        filedmap_file_1 = join(
            Configuration()['directories']['test_data'], 'pipelines', 'phasediff_1.json')
        filedmap_file_2 = join(
            Configuration()['directories']['test_data'], 'pipelines', 'phasediff_2.json')

        pipeline = PipelineTeam98BT()
        test_result = pipeline.get_fieldmap_info(filedmap_file_1, ['magnitude_1', 'magnitude_2'])
        assert test_result[0] == [0.0, 0.0]
        assert test_result[1] == 'magnitude_2'
        test_result = pipeline.get_fieldmap_info(filedmap_file_2, ['magnitude_1', 'magnitude_2'])
        assert test_result[0] == [0.0, 0.0]
        assert test_result[1] == 'magnitude_1'

    @staticmethod
    @mark.pipeline_test
    def test_fieldmap_info():
        """ Test the get_fieldmap_info method """

    @staticmethod
    @mark.pipeline_test
    def test_parameters_files():
        """ Test the get_parameters_files method """

    @staticmethod
    @mark.pipeline_test
    def test_subject_information():
        """ Test the get_subject_information method """

    @staticmethod
    @mark.pipeline_test
    def test_execution():
        """ Test the execution of a PipelineTeam98BT and compare results """
        helpers.test_pipeline_evaluation('98BT')
