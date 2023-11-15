#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_08MQ' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_08MQ.py
    pytest -q test_team_08MQ.py -k <selected_test>
"""

from pytest import helpers, mark
from nipype import Workflow
from nipype.interfaces.base import Bunch

from narps_open.pipelines.team_08MQ import PipelineTeam08MQ

class TestPipelinesTeam08MQ:
    """ A class that contains all the unit tests for the PipelineTeam08MQ class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineTeam08MQ object """

        pipeline = PipelineTeam08MQ()

        # 1 - check the parameters
        assert pipeline.fwhm == 6.0
        assert pipeline.team_id == '08MQ'

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
        """ Test the expected outputs of a PipelineTeam08MQ object """
        pipeline = PipelineTeam08MQ()
        # 1 - 1 subject outputs
        pipeline.subject_list = ['001']
        assert len(pipeline.get_preprocessing_outputs()) == 3*4
        assert len(pipeline.get_run_level_outputs()) == 8+4*3*4
        assert len(pipeline.get_subject_level_outputs()) == 4*3
        assert len(pipeline.get_group_level_outputs()) == 0
        assert len(pipeline.get_hypotheses_outputs()) == 18

        # 2 - 4 subjects outputs
        pipeline.subject_list = ['001', '002', '003', '004']
        assert len(pipeline.get_preprocessing_outputs()) == 3*4*4
        assert len(pipeline.get_run_level_outputs()) == (8+4*3*4)*4
        assert len(pipeline.get_subject_level_outputs()) == 4*3*4
        assert len(pipeline.get_group_level_outputs()) == 0
        assert len(pipeline.get_hypotheses_outputs()) == 18

    @staticmethod
    @mark.unit_test
    def test_subject_information(mocker):
        """ Test the get_subject_information method """

        helpers.mock_event_data(mocker)

        information = PipelineTeam08MQ.get_subject_information('fake_event_file_path')[0]

        assert isinstance(information, Bunch)
        assert information.amplitudes == [[1.0, 1.0], [14.0, 34.0], [6.0, 14.0], [1.0, 1.0]]
        assert information.durations == [[4.0, 4.0], [2.388, 2.289], [2.388, 2.289], [4.0, 4.0]]
        assert information.conditions == ['event', 'gain', 'loss', 'response']
        assert information.onsets == [
        [4.071, 11.834], [4.071, 11.834], [4.071, 11.834], [4.071, 11.834]
        ]

    @staticmethod
    @mark.unit_test
    def test_run_level_contrasts():
        """ Test the get_run_level_contrasts method """

        contrasts = PipelineTeam08MQ.get_run_level_contrasts()
        assert contrasts[0] == ('positive_effect_gain', 'T', ['gain', 'loss'], [1, 0])
        assert contrasts[1] == ('positive_effect_loss', 'T', ['gain', 'loss'], [0, 1])
        assert contrasts[2] == ('negative_effect_loss', 'T', ['gain', 'loss'], [0, -1])

    @staticmethod
    @mark.unit_test
    def test_subgroups_contrasts(mocker):
        """ Test the get_subgroups_contrasts method """

        helpers.mock_participants_data(mocker)

        cei, cer, vei, ver, eii, eri = PipelineTeam08MQ.get_subgroups_contrasts(
            ['sub-001/_contrast_id_1/cope1.nii.gz', 'sub-001/_contrast_id_2/cope1.nii.gz', 'sub-002/_contrast_id_1/cope1.nii.gz', 'sub-002/_contrast_id_2/cope1.nii.gz', 'sub-003/_contrast_id_1/cope1.nii.gz', 'sub-003/_contrast_id_2/cope1.nii.gz', 'sub-004/_contrast_id_1/cope1.nii.gz', 'sub-004/_contrast_id_2/cope1.nii.gz'], # copes
            ['sub-001/_contrast_id_1/varcope1.nii.gz', 'sub-001/_contrast_id_2/varcope1.nii.gz', 'sub-002/_contrast_id_1/varcope1.nii.gz', 'sub-002/_contrast_id_2/varcope1.nii.gz', 'sub-003/_contrast_id_1/varcope1.nii.gz', 'sub-003/_contrast_id_2/varcope1.nii.gz', 'sub-004/_contrast_id_1/varcope1.nii.gz', 'sub-004/_contrast_id_2/varcope1.nii.gz'], # varcopes
            ['001', '002', '003', '004'], # subject_list
            ['fake_participants_file_path'] # participants file
            )

        assert cei == ['sub-001/_contrast_id_1/cope1.nii.gz', 'sub-001/_contrast_id_2/cope1.nii.gz', 'sub-003/_contrast_id_1/cope1.nii.gz', 'sub-003/_contrast_id_2/cope1.nii.gz']
        assert cer == ['sub-002/_contrast_id_1/cope1.nii.gz', 'sub-002/_contrast_id_2/cope1.nii.gz', 'sub-004/_contrast_id_1/cope1.nii.gz', 'sub-004/_contrast_id_2/cope1.nii.gz']
        assert vei == ['sub-001/_contrast_id_1/varcope1.nii.gz', 'sub-001/_contrast_id_2/varcope1.nii.gz', 'sub-003/_contrast_id_1/varcope1.nii.gz', 'sub-003/_contrast_id_2/varcope1.nii.gz']
        assert ver == ['sub-002/_contrast_id_1/varcope1.nii.gz', 'sub-002/_contrast_id_2/varcope1.nii.gz', 'sub-004/_contrast_id_1/varcope1.nii.gz', 'sub-004/_contrast_id_2/varcope1.nii.gz']
        assert eii == ['001', '003']
        assert eri == ['002', '004']

    @staticmethod
    @mark.unit_test
    def test_one_sample_t_test_regressors():
        """ Test the get_one_sample_t_test_regressors method """

        regressors = PipelineTeam08MQ.get_one_sample_t_test_regressors(['001', '002'])
        assert regressors == {'group_mean': [1, 1]}

    @staticmethod
    @mark.unit_test
    def test_two_sample_t_test_regressors():
        """ Test the get_two_sample_t_test_regressors method """

        regressors, groups = PipelineTeam08MQ.get_two_sample_t_test_regressors(
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
        """ Test the execution of a PipelineTeam08MQ and compare results """
        helpers.test_pipeline_evaluation('08MQ')
