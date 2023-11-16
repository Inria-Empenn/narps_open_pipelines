#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.core.common' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_common.py
    pytest -q test_common.py -k <selected_test>
"""
from os import mkdir
from os.path import join, exists, abspath
from shutil import rmtree
from pathlib import Path

from pytest import mark, fixture
from nipype import Node, Function

from narps_open.utils.configuration import Configuration
import narps_open.core.common as co

TEMPORARY_DIR = join(Configuration()['directories']['test_runs'], 'test_common')

@fixture
def remove_test_dir():
    """ A fixture to remove temporary directory created by tests """

    rmtree(TEMPORARY_DIR, ignore_errors = True)
    mkdir(TEMPORARY_DIR)
    yield # test runs here
    rmtree(TEMPORARY_DIR, ignore_errors = True)

class TestCoreCommon:
    """ A class that contains all the unit tests for the common module."""

    @staticmethod
    @mark.unit_test
    def test_remove_file(remove_test_dir):
        """ Test the remove_file function """

        # Create a single file
        test_file_path = abspath(join(TEMPORARY_DIR, 'file1.txt'))
        Path(test_file_path).touch()

        # Check file exist
        assert exists(test_file_path)

        # Create a Nipype Node using remove_files
        test_remove_file_node = Node(Function(
            function = co.remove_file,
            input_names = ['_', 'file_name'],
            output_names = []
            ), name = 'test_remove_file_node')
        test_remove_file_node.inputs._ = ''
        test_remove_file_node.inputs.file_name = test_file_path
        test_remove_file_node.run()

        # Check file is removed
        assert not exists(test_file_path)
