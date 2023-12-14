#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.data.participants' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_participants.py
    pytest -q test_participants.py -k <selected_test>
"""
from os.path import join

from pytest import mark, fixture

import narps_open.data.participants as part
from narps_open.utils.configuration import Configuration

@fixture
def mock_participants_data(mocker):
    """ A fixture to provide mocked data from the test_data directory """

    mocker.patch(
        'narps_open.data.participants.Configuration',
        return_value = {
            'directories': {
                'dataset': join(
                    Configuration()['directories']['test_data'],
                    'data', 'participants')
                }
            }
    )

class TestParticipants:
    """ A class that contains all the unit tests for the participants module."""

    @staticmethod
    @mark.unit_test
    def test_get_participants_information(mock_participants_data):
        """ Test the get_participants_information function """

        p_info = part.get_participants_information()
        assert len(p_info) == 4
        assert p_info.at[1, 'participant_id'] == 'sub-002'
        assert p_info.at[1, 'group'] == 'equalRange'
        assert p_info.at[1, 'gender'] == 'M'
        assert p_info.at[1, 'age'] == 25
        assert p_info.at[2, 'participant_id'] == 'sub-003'
        assert p_info.at[2, 'group'] == 'equalIndifference'
        assert p_info.at[2, 'gender'] == 'F'
        assert p_info.at[2, 'age'] == 27

    @staticmethod
    @mark.unit_test
    def test_get_all_participants():
        """ Test the get_all_participants function """
        participants_list = part.get_all_participants()
        assert len(participants_list) == 108
        assert '001' in participants_list
        assert '105' in participants_list
        assert '123' in participants_list

    @staticmethod
    @mark.unit_test
    def test_get_participants():
        """ Test the get_participants function """

        # Team 2T6S includes all participants
        participants_list = part.get_participants('2T6S')
        assert len(participants_list) == 108

        # Team C88N excludes some participants
        participants_list = part.get_participants('C88N')
        assert len(participants_list) == 106
        assert '001' in participants_list
        assert '076' not in participants_list
        assert '117' not in participants_list

    @staticmethod
    @mark.unit_test
    def test_get_participants_subset():
        """ Test the get_participants_subset function """
        participants_list = part.get_participants_subset()
        assert len(participants_list) == 108
        assert participants_list[0] == '020'
        assert participants_list[-1] == '005'

        participants_list = part.get_participants_subset(20)
        assert len(participants_list) == 20
        assert participants_list[0] == '020'
        assert participants_list[-1] == '087'

        participants_list = part.get_participants_subset(40)
        assert len(participants_list) == 40
        assert participants_list[0] == '020'
        assert participants_list[-1] == '041'

        participants_list = part.get_participants_subset(60)
        assert len(participants_list) == 60
        assert participants_list[0] == '020'
        assert participants_list[-1] == '059'

        participants_list = part.get_participants_subset(80)
        assert len(participants_list) == 80
        assert participants_list[0] == '020'
        assert participants_list[-1] == '003'

    @staticmethod
    @mark.unit_test
    def test_get_group(mock_participants_data):
        """ Test the get_group function """

        assert part.get_group('') == []
        assert part.get_group('equalRange') == ['002', '004']
        assert part.get_group('equalIndifference') == ['001', '003']
