#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_T54A' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_T54A.py
    pytest -q test_team_T54A.py -k <selected_test>
"""

from pytest import helpers, mark
from numpy import isclose
from nipype import Workflow
from nipype.interfaces.base import Bunch

from narps_open.pipelines.team_T54A import PipelineTeamT54A

class TestPipelinesTeamT54A:
    """ A class that contains all the unit tests for the PipelineTeamT54A class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineTeamT54A object """

        pipeline = PipelineTeamT54A()

        # 1 - check the parameters
        assert pipeline.fwhm == 4.0
        assert pipeline.team_id == 'T54A'
        assert pipeline.contrast_list == ['1', '2']
        assert pipeline.run_level_contrasts == [
            ('gain', 'T', ['trial', 'gain', 'loss'], [0, 1, 0]),
            ('loss', 'T', ['trial', 'gain', 'loss'], [0, 0, 1])
            ]

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
        """ Test the expected outputs of a PipelineTeamT54A object """
        pipeline = PipelineTeamT54A()
        # 1 - 1 subject outputs
        pipeline.subject_list = ['001']
        assert len(pipeline.get_preprocessing_outputs()) == 0
        assert len(pipeline.get_run_level_outputs()) == 33*4*1
        assert len(pipeline.get_subject_level_outputs()) == 4*2*1
        assert len(pipeline.get_group_level_outputs()) == 8*2*2 + 4
        assert len(pipeline.get_hypotheses_outputs()) == 18

        # 2 - 4 subjects outputs
        pipeline.subject_list = ['001', '002', '003', '004']
        assert len(pipeline.get_preprocessing_outputs()) == 0
        assert len(pipeline.get_run_level_outputs()) == 33*4*4
        assert len(pipeline.get_subject_level_outputs()) == 4*2*4
        assert len(pipeline.get_group_level_outputs()) == 8*2*2 + 4
        assert len(pipeline.get_hypotheses_outputs()) == 18

    @staticmethod
    @mark.unit_test
    def test_subject_information(mocker):
        """ Test the get_subject_information method """

        helpers.mock_event_data(mocker)

        information = PipelineTeamT54A.get_subject_information('fake_event_file_path')[0]

        assert isinstance(information, Bunch)
        assert information.conditions == [
            'trial',
            'gain',
            'loss',
            'difficulty',
            'response',
            'missed'
            ]

        reference_amplitudes = [
            [1.0, 1.0, 1.0, 1.0],
            [14.0, 34.0, 10.0, 16.0],
            [6.0, 14.0, 15.0, 17.0],
            [1.0, 3.0, 10.0, 9.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0]
        ]
        for reference_array, test_array in zip(reference_amplitudes, information.amplitudes):
            assert isclose(reference_array, test_array).all()

        reference_durations = [
            [2.388, 2.289, 2.08, 2.288],
            [2.388, 2.289, 2.08, 2.288],
            [2.388, 2.289, 2.08, 2.288],
            [2.388, 2.289, 2.08, 2.288],
            [0.0, 0.0, 0.0, 0.0],
            [0.0]
        ]
        for reference_array, test_array in zip(reference_durations, information.durations):
            assert isclose(reference_array, test_array).all()

        reference_onsets = [
            [4.071, 11.834, 27.535, 36.435],
            [4.071, 11.834, 27.535, 36.435],
            [4.071, 11.834, 27.535, 36.435],
            [4.071, 11.834, 27.535, 36.435],
            [6.459, 14.123, 29.615, 38.723],
            [19.535]
        ]
        for reference_array, test_array in zip(reference_onsets, information.onsets):
            assert isclose(reference_array, test_array).all()

    @staticmethod
    @mark.unit_test
    def test_parameters_file(mocker):
        """ Test the get_parameters_file method """

    @staticmethod
    @mark.unit_test
    def test_subgroups_contrasts(mocker):
        """ Test the get_subgroups_contrasts method """

        helpers.mock_participants_data(mocker)

        cei, cer, cg, vei, ver, vg, eii, eri = PipelineTeamT54A.get_subgroups_contrasts(
            ['sub-001/_contrast_id_1/cope1.nii.gz', 'sub-001/_contrast_id_2/cope1.nii.gz', 'sub-002/_contrast_id_1/cope1.nii.gz', 'sub-002/_contrast_id_2/cope1.nii.gz', 'sub-003/_contrast_id_1/cope1.nii.gz', 'sub-003/_contrast_id_2/cope1.nii.gz', 'sub-004/_contrast_id_1/cope1.nii.gz', 'sub-004/_contrast_id_2/cope1.nii.gz'], # copes
            ['sub-001/_contrast_id_1/varcope1.nii.gz', 'sub-001/_contrast_id_2/varcope1.nii.gz', 'sub-002/_contrast_id_1/varcope1.nii.gz', 'sub-002/_contrast_id_2/varcope1.nii.gz', 'sub-003/_contrast_id_1/varcope1.nii.gz', 'sub-003/_contrast_id_2/varcope1.nii.gz', 'sub-004/_contrast_id_1/varcope1.nii.gz', 'sub-004/_contrast_id_2/varcope1.nii.gz'], # varcopes
            ['001', '002', '003', '004'], # subject_list
            ['fake_participants_file_path'] # participants file
            )

        assert cei == ['sub-001/_contrast_id_1/cope1.nii.gz', 'sub-001/_contrast_id_2/cope1.nii.gz', 'sub-003/_contrast_id_1/cope1.nii.gz', 'sub-003/_contrast_id_2/cope1.nii.gz']
        assert cer == ['sub-002/_contrast_id_1/cope1.nii.gz', 'sub-002/_contrast_id_2/cope1.nii.gz', 'sub-004/_contrast_id_1/cope1.nii.gz', 'sub-004/_contrast_id_2/cope1.nii.gz']
        assert cg ==  ['sub-001/_contrast_id_1/cope1.nii.gz', 'sub-001/_contrast_id_2/cope1.nii.gz', 'sub-002/_contrast_id_1/cope1.nii.gz', 'sub-002/_contrast_id_2/cope1.nii.gz', 'sub-003/_contrast_id_1/cope1.nii.gz', 'sub-003/_contrast_id_2/cope1.nii.gz', 'sub-004/_contrast_id_1/cope1.nii.gz', 'sub-004/_contrast_id_2/cope1.nii.gz']
        assert vei == ['sub-001/_contrast_id_1/varcope1.nii.gz', 'sub-001/_contrast_id_2/varcope1.nii.gz', 'sub-003/_contrast_id_1/varcope1.nii.gz', 'sub-003/_contrast_id_2/varcope1.nii.gz']
        assert ver == ['sub-002/_contrast_id_1/varcope1.nii.gz', 'sub-002/_contrast_id_2/varcope1.nii.gz', 'sub-004/_contrast_id_1/varcope1.nii.gz', 'sub-004/_contrast_id_2/varcope1.nii.gz']
        assert vg == ['sub-001/_contrast_id_1/varcope1.nii.gz', 'sub-001/_contrast_id_2/varcope1.nii.gz', 'sub-002/_contrast_id_1/varcope1.nii.gz', 'sub-002/_contrast_id_2/varcope1.nii.gz', 'sub-003/_contrast_id_1/varcope1.nii.gz', 'sub-003/_contrast_id_2/varcope1.nii.gz', 'sub-004/_contrast_id_1/varcope1.nii.gz', 'sub-004/_contrast_id_2/varcope1.nii.gz']
        assert eii == ['001', '003']
        assert eri == ['002', '004']

    @staticmethod
    @mark.unit_test
    def test_one_sample_t_test_regressors():
        """ Test the get_one_sample_t_test_regressors method """

        regressors = PipelineTeamT54A.get_one_sample_t_test_regressors(['001', '002'])
        assert regressors == {'group_mean': [1, 1]}

    @staticmethod
    @mark.unit_test
    def test_two_sample_t_test_regressors():
        """ Test the get_two_sample_t_test_regressors method """

        regressors, groups = PipelineTeamT54A.get_two_sample_t_test_regressors(
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
        """ Test the execution of a PipelineTeamT54A and compare results """
        helpers.test_pipeline_evaluation('T54A')
