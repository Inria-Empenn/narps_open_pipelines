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
from shutil import rmtree, move, copytree
from time import sleep

from checksumdir import dirhash
from pytest import mark

from narps_open.data.results import ResultsCollection, ResultsCollectionFactory
from narps_open.data.results.team_2T6S import ResultsCollection2T6S
from narps_open.utils import hash_dir_images
from narps_open.utils.configuration import Configuration

class TestResultsCollectionFactory:
    """ A class that contains all the unit tests for the ResultsCollectionFactory class."""

    @staticmethod
    @mark.unit_test
    def test_get_collection():
        """ Test the get_collection of a ResultsCollectionFactory object """

        factory = ResultsCollectionFactory()
        assert isinstance(factory.get_collection('2T6S'), ResultsCollection2T6S)
        assert isinstance(factory.get_collection('Q6O0'), ResultsCollection)

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
        assert collection.files['hypo1_thresh.nii.gz'] == test_str

        collection = ResultsCollection('43FJ')
        assert collection.team_id == '43FJ'
        assert collection.uid == '4824'
        assert 'results/orig/4824_43FJ' in collection.directory
        test_str = 'http://neurovault.org/media/images/4824/'
        test_str += 'Zstat_Thresholded_Negative_Effect_of_Loss_Equal_Indifference.nii.gz'
        assert collection.files['hypo5_thresh.nii.gz'] == test_str

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

        # Mock the results path
        results_directory = Configuration()['directories']['narps_results']
        Configuration()['directories']['narps_results'] = Configuration()['directories']['test_runs']

        # Init & download the collection
        collection = ResultsCollection('2T6S')
        assert collection.uid == '15001'
        collection.download()

        # Collection should be downloaded in the results directory
        expected_dir = join(Configuration()['directories']['narps_results'], 'orig', '15001_2T6S')

        # Test presence of the download
        assert isdir(expected_dir)
        assert dirhash(expected_dir) == '4c6a53c163e033d62e9728acd62b12ee'

        # Write back the results path in configuration
        Configuration()['directories']['narps_results'] = results_directory

        # Remove folder
        rmtree(expected_dir)

class TestResultsCollection2T6S:
    """ A class that contains all the unit tests for the ResultsCollection2T6S class."""

    @staticmethod
    @mark.unit_test
    def test_rectify():
        """ Test the rectify method """

        # Get raw data
        orig_directory = join(
            Configuration()['directories']['test_data'], 'data', 'results', 'team_2T6S')

        # Create test data
        test_directory = join(Configuration()['directories']['test_runs'], 'results_team_2T6S')
        if isdir(test_directory):
            rmtree(test_directory)

        # Init ResultsCollection
        collection = ResultsCollection2T6S()
        collection.directory = test_directory
        copytree(orig_directory, test_directory)

        # Test copy
        assert dirhash(test_directory) == 'fa9fedc73f575d322e15d8516eee9da9'

        # Rectify data
        collection.rectify()

        # Check rectification
        value = '4b83f985657f4fe4bb06cf42b5214b597dd8b99f1b753ac3183b54583bb3be16'
        assert hash_dir_images(test_directory) == value

        # Delete data
        rmtree(test_directory)
