#!/usr/bin/python
# coding: utf-8

""" Allow access to the configuration for the narps_open project. """

from os.path import join
from importlib_resources import files

from tomli import load

from narps_open.utils.singleton import SingletonMeta

class Configuration(dict, metaclass=SingletonMeta):
    """ This class allows to access project configuration files """

    def __init__(self, config_type: str = 'default'):
        super().__init__()

        # Set configuration type
        self._config_type = config_type

        # Select configuration file from config_type
        file_name = ''
        self._project_config_path = files('narps_open.utils.configuration')
        if self._config_type == 'default':
            file_name = join(self._project_config_path, 'default_config.toml')
        elif self._config_type == 'testing':
            file_name = join(self._project_config_path, 'testing_config.toml')
        elif self._config_type == 'custom':
            # a configuration file must be passed at execution time
            pass
        else:
            raise AttributeError(f'Unknown configuration type: {self._config_type}')

        if file_name != '':
            # It is important to use the property setter here, so that the dict self is updated
            self.config_file = file_name

    @property
    def config_type(self) -> str:
        """ Getter for property config_type """
        return self._config_type

    @config_type.setter
    def config_type(self, value: str) -> None:
        """ Setter for property config_type """
        self._config_type = value

    @property
    def config_file(self) -> str:
        """ Getter for property config_file """
        return self._config_file

    @config_file.setter
    def config_file(self, value: str) -> None:
        """ Setter for property config_file """
        self._config_file = value

        # Update configuration using this new file
        with open(self._config_file, 'rb') as file:
            self.clear()
            self.update(load(file))
