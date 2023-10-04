#!/usr/bin/python
# coding: utf-8

""" A mdoule to parse task data from NARPS for the narps_open package """

from os.path import join
from json import load

from narps_open.utils.configuration import Configuration
from narps_open.utils.singleton import SingletonMeta

class TaskInformation(dict, metaclass=SingletonMeta):
    """ This class allows to access information about the task performed in NARPS """

    def __init__(self):
        super().__init__()

        # Load information from the task-MGT_bold.json file
        file_name = join(Configuration()['directories']['dataset'], 'task-MGT_bold.json')
        self.update(load(file))

        # Compute derived information
        self['NumberOfSlices'] = len(self['SliceTiming'])
        self['AcquisitionTime'] = self['RepetitionTime'] / self['NumberOfSlices']
        self['TotalReadoutTime'] = self['NumberOfSlices'] * self['EffectiveEchoSpacing']
