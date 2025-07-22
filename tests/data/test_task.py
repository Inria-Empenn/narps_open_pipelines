#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.data.task' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_task.py
    pytest -q test_task.py -k <selected_test>
"""
from os.path import join

from pytest import mark, fixture

from narps_open.utils.configuration import Configuration
from narps_open.data import task

@fixture(scope='function', autouse=True)
def load_test_task_data():
    """ Init the TaskInformation class with task data for testing """
    task.TaskInformation(
        join(Configuration()['directories']['test_data'], 'data', 'task', 'task-info.json')
        )

class TestTaskInformation:
    """ A class that contains all the unit tests for the TaskInformation class."""

    @staticmethod
    @mark.unit_test
    def test_accessing():
        """ Check that task information is reachable """

        assert task.TaskInformation()['RepetitionTime'] == 1
        assert len(task.TaskInformation()['SliceTiming']) == 6

    @staticmethod
    @mark.unit_test
    def test_singleton():
        """ Check that TaskInformation is a singleton. """

        obj1 = task.TaskInformation()
        obj2 = task.TaskInformation()

        assert id(obj1) == id(obj2)

    @staticmethod
    @mark.unit_test
    def test_derived():
        """ Test the derived values of a TaskInformation object """

        task_info = task.TaskInformation()
        assert task_info['NumberOfSlices'] == 6
        assert task_info['AcquisitionTime'] == 1 / 6
        assert task_info['TotalReadoutTime'] == 12
