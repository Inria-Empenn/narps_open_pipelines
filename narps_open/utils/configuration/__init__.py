#!/usr/bin/python
# coding: utf-8

""" Allow access to the configuration for the narps_open project. """

from os import path
from configparser import ConfigParser, ExtendedInterpolation
from importlib_resources import files

from narps_open.utils.singleton import SingletonMeta

class Configuration(metaclass=SingletonMeta):
    """ This class allows to access project configuration files """

    def __init__(self):
        super().__init__()

        # Import configuration
        self._configuration = ConfigParser(interpolation=ExtendedInterpolation())
        self._configuration_type = config_type

        # Select configuration file
        file_name = ''
        if self._configuration_type == 'testing':
            file_name = 'test_config.ini'
        elif self._configuration_type == 'preproduction':
            file_name = 'preprod_config.ini'
        elif self._configuration_type == 'production':
            file_name = 'prod_config.ini'
        else:
            raise AttributeError(f'{self._configuration_type} is not a valid configuration type.')

        # Parse configuration file
        file = files('narps_open.utils.configuration').joinpath(file_name)
        self._configuration.read(file, encoding = 'utf8')

    def get(self, *args, **kwargs):
        """ Using ConfigParser's get() method """
        return self._configuration.get(*args, **kwargs)

    def getboolean(self, *args, **kwargs):
        """ Using ConfigParser's getboolean() method """
        return self._configuration.getboolean(*args, **kwargs)
