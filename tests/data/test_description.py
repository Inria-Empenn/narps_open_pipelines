#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.data.description' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_description.py
    pytest -q test_description.py -k <selected_test>
"""

from os.path import join

from pytest import raises, mark

from narps_open.utils.configuration import Configuration
from narps_open.data.description import TeamDescription

class TestUtilsDescription:
    """ A class that contains all the unit tests for the description module."""

    @staticmethod
    @mark.unit_test
    def test_creation():
        """ Test the creation of a TeamDescription object """

        # Instantiation with no parameters
        with raises(TypeError):
            TeamDescription()

        # Instantiation with wrong team id
        with raises(AttributeError):
            TeamDescription('wrong_id')

        # Instantiation is ok
        assert TeamDescription('2T6S') is not None

    @staticmethod
    @mark.unit_test
    def test_arguments_properties():
        """ Test the arguments and properties of a TeamDescription object """

        # 1 - Create a TeamDescription
        description = TeamDescription('2T6S')

        # 2 - Check arguments
        assert description.team_id == '2T6S'

        # 3 - Check access as dict
        assert description['general.softwares'] == 'SPM12 , \nfmriprep 1.1.4'
        assert description['exclusions.n_participants'] == '108'
        assert description['preprocessing.motion'] == '6'
        assert description['analysis.RT_modeling'] == 'duration'
        assert description['categorized_for_analysis.analysis_SW_with_version'] == 'SPM12'
        assert description['derived.func_fwhm'] == '8'
        assert description['comments.excluded_from_narps_analysis'] == 'No'

        # 4 - Check properties
        assert isinstance(description.general, dict)
        assert isinstance(description.exclusions, dict)
        assert isinstance(description.preprocessing, dict)
        assert isinstance(description.analysis, dict)
        assert isinstance(description.categorized_for_analysis, dict)
        assert isinstance(description.derived, dict)
        assert isinstance(description.comments, dict)

        assert list(description.general.keys()) == [
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
        assert description.general['softwares'] == 'SPM12 , \nfmriprep 1.1.4'
        assert description.exclusions['n_participants'] == '108'
        assert description.preprocessing['motion'] == '6'
        assert description.analysis['RT_modeling'] == 'duration'
        assert description.categorized_for_analysis['analysis_SW_with_version'] == 'SPM12'
        assert description.derived['func_fwhm'] == '8'
        assert description.comments['excluded_from_narps_analysis'] == 'No'

        # 6 - Test another team
        description = TeamDescription('9Q6R')
        assert description.team_id == '9Q6R'
        assert description['general.softwares'] == 'FSL 5.0.11, MRIQC, FMRIPREP'
        assert isinstance(description.general, dict)
        assert description.general['softwares'] == 'FSL 5.0.11, MRIQC, FMRIPREP'

    @staticmethod
    @mark.unit_test
    def test_markdown():
        """ Test writing a TeamDescription as Markdown """

        # Generate markdown from description
        description = TeamDescription('9Q6R')
        markdown = description.markdown()

        # Compare markdown with test file
        test_file_path = join(
            Configuration()['directories']['test_data'],
            'data', 'description', 'test_markdown.md'
            )
        with open(test_file_path, 'r', encoding = 'utf-8') as file:
            assert markdown == file.read()

    @staticmethod
    @mark.unit_test
    def test_str():
        """ Test writing a TeamDescription as JSON """

        # Generate report
        description = TeamDescription('9Q6R')

        # Compare string version of the description with test file
        test_file_path = join(
            Configuration()['directories']['test_data'],
            'data', 'description', 'test_str.json'
            )
        with open(test_file_path, 'r', encoding = 'utf-8') as file:
            assert str(description) == file.read()
