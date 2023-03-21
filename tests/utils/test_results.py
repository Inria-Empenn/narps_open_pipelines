#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.utils.results' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_results.py
    pytest -q test_results.py -k <selected_test>
"""

from pytest import raises, mark

from narps_open.utils.results import show_progress, download_result_collection

class TestUtilsResults:
    """ A class that contains all the unit tests for the results module."""

    @staticmethod
    @mark.unit_test
    def test_show_progress(capfd): # using pytest's capfd fixture to get stdout
        """ Test the show_progress function """

        show_progress(25,1,100)
        captured = capfd.readouterr()
        assert captured.out == 'Downloading 25 %\r'

        show_progress(26,2,200)
        captured = capfd.readouterr()
        assert captured.out == 'Downloading 26 %\r'

        show_progress(25,50,-1)
        captured = capfd.readouterr()
        assert captured.out == 'Downloading ⣽\r'

    @staticmethod
    @mark.unit_test
    def test_download_result_collection():
        """ Test the download_result_collection function """
