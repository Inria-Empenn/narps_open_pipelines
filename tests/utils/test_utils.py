#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.utils' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_utils.py
    pytest -q test_utils.py -k <selected_test>
"""

from narps_open.utils import get_all_participants, get_participants

class TestUtils:
    """ A class that contains all the unit tests for the utils module."""

    @staticmethod
    def test_get_all_participants():
        """ Test the get_all_participants function """
        participants_list = get_all_participants()
        assert len(participants_list) == 108
        assert '001' in participants_list
        assert '105' in participants_list
        assert '123' in participants_list

    @staticmethod
    def test_get_participants():
        """ Test the get_participants function """

        # Team 2T6S includes all participants
        participants_list = get_participants('2T6S')
        assert len(participants_list) == 108

        # Team C88N excludes some participants
        participants_list = get_participants('C88N')
        assert len(participants_list) == 106
        assert '001' in participants_list
        assert '076' not in participants_list
        assert '117' not in participants_list
