#!/usr/bin/python
# coding: utf-8

""" Interfaces that declare operations and parameters common to all pipelines. """

from os.path import join, abspath
from random import choices
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from importlib_resources import files

class PipelineDirectories():
    """ This object contains paths to the directories of interest for a Pipeline """

    def __init__(self):
        """ Attributes (properties) of class PipelineDirectories are:
            - dataset_dir: str, directory where the ds001734 dataset is stored
            (same for each inheriting pipeline)
            - working_dir: str, directory where to store intermediate results
            (will change for each inheriting pipeline)
            - output_dir: str, directory for final results
            (will change for each inheriting pipeline)

            These 3 directories are based on the following 'private' attributes:
            - _root_dir: str, absolute path of the root directory of the project
            - _results_dir: str, base directory for results to be stored

        """
        # Find the root directory using importlib_resources' files()
        self._root_dir = abspath(files('narps_open').joinpath('..'))
        self._results_dir = join(self._root_dir, 'data', 'derived', 'reproduced')

        # Initialize other directories
        self._dataset_dir = join(self._root_dir, 'data', 'original', 'ds001734')
        self._working_dir = ''
        self._output_dir = ''

    @property
    def dataset_dir(self):
        """ Getter for property exp_dir """
        return self._dataset_dir

    @property
    def working_dir(self):
        """ Getter for property working_dir """
        return self._working_dir

    @working_dir.setter
    def working_dir(self, dir_name):
        """ Setter for property working_dir """
        self._working_dir = dir_name

    @property
    def output_dir(self):
        """ Getter for property output_dir """
        return self._output_dir

    @output_dir.setter
    def output_dir(self, dir_name):
        """ Setter for property output_dir """
        self._output_dir = dir_name

    def setup(self, team_id: str = 'None') -> None:
        """ Set the directories according to the team ID """
        self._working_dir = join(
            self._results_dir,f'NARPS-{team_id}-reproduced/intermediate_results')
        self._output_dir = join(self._results_dir, f'NARPS-{team_id}-reproduced')

class Pipeline(ABC):
    """ An abstract class from which pipelines can inherit. """

    @abstractmethod
    def __init__(self):
        """ Attributes of class Pipeline are:
        # Directories
            - directories: PipelineDirectories, an object that contains all the paths of interest
            for the pipeline

        # Lists of data
            - subject_list: list of str, list of subject for which you want to do the analysis
            - run_list: list of str, list of runs for which you want to do the analysis

        # Parameters
            - fwhm: float, Full width at half maximum (in mm) for the Gaussian smoothing kernel
            - tr: float, time repetition (in s) used during acquisition. This value is the same
            for all pipelines as they use the same dataset.
        """
        # Directories
        self._directories = PipelineDirectories()

        # Data
        self._subject_list = []
        self._run_list = []

        # Parameters
        self._fwhm = 0.0
        self._tr = 1.0

    @property
    def directories(self):
        """ Getter for property directories """
        return self._directories

    @directories.setter
    def directories(self, value: PipelineDirectories):
        """ Setter for property fwhm """
        self._directories = value

    @property
    def tr(self):
        """ Getter for property tr """
        return self._tr

    @property
    def fwhm(self):
        """ Getter for property fwhm """
        return self._fwhm

    @fwhm.setter
    def fwhm(self, value):
        """ Setter for property fwhm """
        self._fwhm = value

    @abstractmethod
    def get_preprocessing(self):
        """ Return a Nipype worflow describing the prerpocessing part of the pipeline """

    @abstractmethod
    def get_subject_level_analysis(self):
        """ Return a Nipype worflow describing the subject level analysis part of the workflow """

    @abstractmethod
    def get_group_level_analysis(self):
        """ Return a Nipype worflow describing the group level analysis part of the workflow """
