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

        # Define another child for NodeCreator
        class ValidNC(nd.NodeCreator):
            def create_node(name: str) -> Node:
                return Node(Select(), name = name)

        # Test it can be instantiated
        test_node = ValidNC.create_node('node_name')
        assert isinstance(test_node, Node)
        assert isinstance(test_node.interface, Select)
        assert test_node.name == 'node_name'

class TestRemoveDirectoryNodeCreator:
    """ A class that contains all the unit tests for the RemoveDirectoryNodeCreator class."""

    @staticmethod
    @mark.unit_test
    def test_create_node():
        """ Test the create_node method """

        test_node = nd.RemoveDirectoryNodeCreator.create_node('node_name')
        assert isinstance(test_node, Node)
        assert isinstance(test_node.interface, Function)
        assert test_node.name == 'node_name'

class TestRemoveFileNodeCreator:
    """ A class that contains all the unit tests for the RemoveFileNodeCreator class."""

    @staticmethod
    @mark.unit_test
    def test_create_node():
        """ Test the create_node method """

        test_node = nd.RemoveFileNodeCreator.create_node('node_name')
        assert isinstance(test_node, Node)
        assert isinstance(test_node.interface, Function)
        assert test_node.name == 'node_name'
