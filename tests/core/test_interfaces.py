#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.core.interfaces' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_interfaces.py
    pytest -q test_interfaces.py -k <selected_test>
"""

from pytest import mark

from nipype.interfaces.base.core import Interface
from nipype.interfaces.utility import Select, Function

from narps_open.core import interfaces

class ValidNC(interfaces.InterfaceCreator):
    """ A valid implementation of a InterfaceCreator, for test purposes """

    @staticmethod
    def create_interface() -> Interface:
        """ Return a Interface, as expected """
        return Select()

class TestInterfaceCreator:
    """ A class that contains all the unit tests for the InterfaceCreator class."""

    @staticmethod
    @mark.unit_test
    def test_create_interface():
        """ Test the create_interface method """

        test_node = ValidNC.create_interface()
        assert isinstance(test_node, Select)

class TestRemoveParentDirectoryInterfaceCreator:
    """ A class that contains all the unit tests for the
        RemoveParentDirectoryInterfaceCreator class.
    """

    @staticmethod
    @mark.unit_test
    def test_create_interface():
        """ Test the create_interface method """

        test_node = interfaces.RemoveParentDirectoryInterfaceCreator.create_interface()
        assert isinstance(test_node, Function)
        inputs = str(test_node.inputs)
        assert '_ = <undefined>' in inputs
        assert 'file_name = <undefined>' in inputs
        assert 'function_str = def remove_parent_directory(_, file_name: str) -> None:' in inputs

class TestRemoveDirectoryInterfaceCreator:
    """ A class that contains all the unit tests for the RemoveDirectoryInterfaceCreator class."""

    @staticmethod
    @mark.unit_test
    def test_create_interface():
        """ Test the create_interface method """

        test_node = interfaces.RemoveDirectoryInterfaceCreator.create_interface()
        assert isinstance(test_node, Function)
        inputs = str(test_node.inputs)
        assert '_ = <undefined>' in inputs
        assert 'directory_name = <undefined>' in inputs
        assert 'function_str = def remove_directory(_, directory_name: str) -> None:' in inputs

class TestRemoveFileInterfaceCreator:
    """ A class that contains all the unit tests for the RemoveFileInterfaceCreator class."""

    @staticmethod
    @mark.unit_test
    def test_create_interface():
        """ Test the create_interface method """

        test_node = interfaces.RemoveFileInterfaceCreator.create_interface()
        assert isinstance(test_node, Function)
        inputs = str(test_node.inputs)
        assert '_ = <undefined>' in inputs
        assert 'file_name = <undefined>' in inputs
        assert 'function_str = def remove_file(_, file_name: str) -> None:' in inputs
