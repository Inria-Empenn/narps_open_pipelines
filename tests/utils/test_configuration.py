#!/usr/bin/python
# coding: utf-8

"""Tests of the narps_open.utils.configuration module.

Launch this test with PyTest

Usage:
======
    pytest -q test_configuration.py
"""

from importlib import reload
from pytest import fixture, mark, raises

import narps_open.utils.configuration as cfg

@fixture(scope='function', autouse=True)
def reload_module():
    """ Reload module narps_open.utils.configuration between each test
        so that the Configuration singleton is reset.
    """
    reload(cfg)

@fixture(scope='function', autouse=False)
def get_testing_configuration():
    """ Get a copy of the testing configuration, to be able to access it
        while manipulating other configuration types.
    """
    config = cfg.Configuration('testing')
    reload(cfg)

    return config

class TestConfiguration():
    """ A class that contains all the unit tests."""
    # TODO test cases where configuration files are not found / loaded

    @staticmethod
    @mark.unit_test
    def test_accessing():
        """ Check that configuration is reachable """

        cfg.Configuration(config_type='testing')
        assert cfg.Configuration()['general']['config_type'] == 'testing'

    @staticmethod
    @mark.unit_test
    def test_singleton():
        """ Check that configuration type is set at a class level by the
        first instance."""

        obj1 = cfg.Configuration(config_type='testing')
        obj2 = cfg.Configuration(config_type='default')

        assert obj1.config_type == 'testing'
        assert obj1['general']['config_type'] == 'testing'
        assert obj2.config_type == 'testing'
        assert obj2['general']['config_type'] == 'testing'

    @staticmethod
    @mark.unit_test
    def test_unknown_config_type():
        """ Check loading config with an unknown type raises an exception."""

        with raises(AttributeError):
            cfg.Configuration(config_type='wrong_type')

    @staticmethod
    @mark.unit_test
    def test_defaults():
        """ Check loading default config by default."""

        obj1 = cfg.Configuration()
        assert obj1.config_type == 'default'
        assert obj1['general']['config_type'] == 'default'

    @staticmethod
    @mark.unit_test
    def test_custom_wrong_case_1(get_testing_configuration):
        """ Check loading custom config file.
            Error case : trying to load a custom file, not in custom mode
        """
        # Get test_data from testing_configuration (loaded by the get_testing_configuration fixture
        test_data_dir = get_testing_configuration['directories']['test_data']

        obj1 = cfg.Configuration('default')
        with raises(AttributeError):
            obj1.config_file = test_data_dir+'utils/configuration/custom_config.toml'

    @staticmethod
    @mark.unit_test
    def test_custom_wrong_case_3():
        """ Check loading custom config file.
            Error case : trying to load a custom file that doesn't exist
        """
        obj1 = cfg.Configuration('custom')
        with raises(FileNotFoundError):
            obj1.config_file = '/path/to/custom/config_file'

    @staticmethod
    @mark.unit_test
    def test_custom(get_testing_configuration):
        """ Check loading custom config file."""

        # Get test_data from testing_configuration (loaded by the get_testing_configuration fixture
        test_data_dir = get_testing_configuration['directories']['test_data']

        # Load custom file
        obj1 = cfg.Configuration('custom')
        obj1.config_file = test_data_dir+'utils/configuration/custom_config.toml'

        assert obj1['general']['config_type'] == 'custom'
        assert obj1['tests']['parameter_1'] == 125
        assert obj1['tests']['parameter_2'] == 'value'
