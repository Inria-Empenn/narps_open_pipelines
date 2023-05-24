#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.data.participants' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_participants.py
    pytest -q test_participants.py -k <selected_test>
"""

from pytest import mark

import narps_open.data.participants as part

class TestParticipants:
    """ A class that contains all the unit tests for the participants module."""

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
