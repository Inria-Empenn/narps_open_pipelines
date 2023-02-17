#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.utils.configuration' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_configuration.py
    pytest -q test_configuration.py -k <selected_test>
"""

from pytest import raises

from narps_open.utils.configuration import TeamConfiguration

class TestUtilsConfiguration:
    """ A class that contains all the unit tests for the configuration module."""

    @staticmethod
    def test_creation():
        """ Test the creation of a TeamConfiguration object """

        # Instantiation with no parameters
        with raises(TypeError):
            TeamConfiguration()

        # Instantiation with wrong team id
        with raises(AttributeError):
            TeamConfiguration('wrong_id')

        # Instatiation is ok
        assert TeamConfiguration('2T6S') is not None

    @staticmethod
    def test_arguments_properties():
        """ Test the arguments and properties of a TeamConfiguration object """

        # 1 - Create a TeamConfiguration
        configuration = TeamConfiguration('2T6S')

        # 2 - Check arguments
        assert configuration.team_id == '2T6S'

        # 3 - Check access as dict
        assert configuration['general.softwares'] == 'SPM12 , \nfmriprep 1.1.4'
        assert configuration['exclusions.n_participants'] == '108'
        assert configuration['preprocessing.motion'] == '6'
        assert configuration['analysis.RT_modeling'] == 'duration'
        assert configuration['categorized_for_analysis.analysis_SW_with_version'] == 'SPM12'

        # 4 - Check properties
        assert isinstance(configuration.general, dict)
        assert isinstance(configuration.exclusions, dict)
        assert isinstance(configuration.preprocessing, dict)
        assert isinstance(configuration.analysis, dict)
        assert isinstance(configuration.categorized_for_analysis, dict)

        assert list(configuration.general.keys()) == [
            'teamID',
            'NV_collection_link',
            'results_comments',
            'preregistered',
            'link_preregistration_form',
            'regions_definition',
            'softwares',
            'general_comments'
            ]

        # 5 - Check properties values
        assert configuration.general['softwares'] == 'SPM12 , \nfmriprep 1.1.4'
        assert configuration.exclusions['n_participants'] == '108'
        assert configuration.preprocessing['motion'] == '6'
        assert configuration.analysis['RT_modeling'] == 'duration'
        assert configuration.categorized_for_analysis['analysis_SW_with_version'] == 'SPM12'

        # 6 - Test another team
        configuration = TeamConfiguration('9Q6R')
        assert configuration.team_id == '9Q6R'
        assert configuration['general.softwares'] == 'FSL 5.0.11, MRIQC, FMRIPREP'
        assert isinstance(configuration.general, dict)
        assert configuration.general['softwares'] == 'FSL 5.0.11, MRIQC, FMRIPREP'
