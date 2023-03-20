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
