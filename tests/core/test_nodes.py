#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.core.nodes' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_nodes.py
    pytest -q test_nodes.py -k <selected_test>
"""

from pytest import mark, raises

from nipype import Node
from nipype.interfaces.utility import Select, Function

import narps_open.core.nodes as nd
from narps_open.core.common import remove_directory, remove_file

class TestNodeCreator:
    """ A class that contains all the unit tests for the NodeCreator class."""

    @staticmethod
    @mark.unit_test
    def test_create_node():
        """ Test the create_node method """

        # It is not possible to create an instance of a NodeCreator
        with raises(Exception):
            nd.NodeCreator().create_node('node_name')

        # Define a child for NodeCreator
        class ErrorNC(nd.NodeCreator):
            def random_method(self):
                pass

        # Test it cannot be instanciated
        with raises(Exception):
            ErrorNC().create_node('node_name')

        # Define another child for NodeCreator
        class ValidNC(nd.NodeCreator):
            def create_node(self, name: str) -> Node:
                return Node(Select(), name = name)

        # Test it can be instanciated
        test_node = ValidNC().create_node('node_name')
        assert isinstance(test_node, Node)
        assert isinstance(test_node.interface, Select)
        assert test_node.name == 'node_name'

class TestRemoveDirectoryNodeCreator:
    """ A class that contains all the unit tests for the RemoveDirectoryNodeCreator class."""

    @staticmethod
    @mark.unit_test
    def test_create_node():
        """ Test the create_node method """

        test_node = nd.RemoveDirectoryNodeCreator().create_node('node_name')
        assert isinstance(test_node, Node)
        assert isinstance(test_node.interface, Function)
        assert test_node.name == 'node_name'

class TestRemoveFileNodeCreator:
    """ A class that contains all the unit tests for the RemoveFileNodeCreator class."""

    @staticmethod
    @mark.unit_test
    def test_create_node():
        """ Test the create_node method """

        test_node = nd.RemoveFileNodeCreator().create_node('node_name')
        assert isinstance(test_node, Node)
        assert isinstance(test_node.interface, Function)
        assert test_node.name == 'node_name'
