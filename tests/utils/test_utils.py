#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.utils' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_utils.py
    pytest -q test_utils.py -k <selected_test>
"""

from pytest import mark

from narps_open.utils import show_download_progress

class TestUtils:
    """ A class that contains all the unit tests for the utils module."""

    @staticmethod
    @mark.unit_test
    def test_show_download_progress(capfd): # using pytest's capfd fixture to get stdout
        """ Test the show_download_progress function """

        show_download_progress(25,1,100)
        captured = capfd.readouterr()
        assert captured.out == 'Downloading 25 %\r'

        show_download_progress(26,2,200)
        captured = capfd.readouterr()
        assert captured.out == 'Downloading 26 %\r'

        show_download_progress(25,50,-1)
        captured = capfd.readouterr()
        assert captured.out == 'Downloading ⣽\r'
