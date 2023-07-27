#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.data.results' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_results.py
    pytest -q test_results.py -k <selected_test>
"""

from os.path import isdir, join
from shutil import rmtree

from checksumdir import dirhash
from pytest import mark

from narps_open.data.results import ResultsCollection
from narps_open.utils.configuration import Configuration

class TestResultsCollection:
    """ A class that contains all the unit tests for the ResultsCollection class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a ResultsCollection object """

        collection = ResultsCollection('2T6S')
        assert collection.team_id == '2T6S'
        assert collection.uid == '4881'
        assert 'results/orig/4881_2T6S' in collection.directory
        test_str = 'http://neurovault.org/media/images/4881/hypo1_thresholded_revised.nii.gz'
        assert collection.files['hypo1_thresh'] == test_str

        collection = ResultsCollection('43FJ')
        assert collection.team_id == '43FJ'
        assert collection.uid == '4824'
        assert 'results/orig/4824_43FJ' in collection.directory
        test_str = 'http://neurovault.org/media/images/4824/'
        test_str += 'Zstat_Thresholded_Negative_Effect_of_Loss_Equal_Indifference.nii.gz'
        assert collection.files['hypo5_thresh'] == test_str

    @staticmethod
    @mark.unit_test
    def test_download(mocker):
        """ Test the download method

        This method uses the mocker from pytest-mock to replace `ResultsCollection.get_uid`,
        in order to provide a fake Neurovault collection uid.
        """

        # Mock the get_uid method
        def fake_get_uid(_):
            return '15001'
        mocker.patch.object(ResultsCollection, 'get_uid', fake_get_uid)

        # Init & download the collection
        collection = ResultsCollection('2T6S')
        assert collection.uid == '15001'
        collection.download()

        # Collection should be downloaded in the results directory
        expected_dir = join(Configuration()['directories']['narps_results'], 'orig', '15001_2T6S')

        # Test presence of the download
        assert isdir(expected_dir)
        assert dirhash(expected_dir) == '4c6a53c163e033d62e9728acd62b12ee'

        # Remove folder
        rmtree(expected_dir)
