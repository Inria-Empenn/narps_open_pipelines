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
        assert collection.id == '4881'
        assert collection.url == 'https://neurovault.org/collections/4881/download'
        assert 'results/orig/4881_2T6S' in collection.directory
        assert collection.result_names['hypo3_thresh.nii.gz'] == 'hypo3_thresholded_revised.nii.gz'

        collection = ResultsCollection('C88N')
        assert collection.team_id == 'C88N'
        assert collection.id == '4812'
        assert collection.url == 'https://neurovault.org/collections/4812/download'
        assert 'results/orig/4812_C88N' in collection.directory
        assert collection.result_names['hypo3_thresh.nii.gz'] == 'hypo3_thresh.nii.gz'

    @staticmethod
    @mark.unit_test
    def test_download():
        """ Test the download method """

        ResultsCollection('2T6S').download()

        # Collection should be downloaded in the results directory
        expected_dir = join(Configuration()['directories']['narps_results'], 'orig', '4881_2T6S')

        # Test presence of the download
        assert isdir(expected_dir)
        assert dirhash(expected_dir) == '26af20dc7847bcb14d4452239ea458e8'

        # Remove folder
        rmtree(expected_dir)
