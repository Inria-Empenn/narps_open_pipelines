#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.core.nodes' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_nodes.py
    pytest -q test_nodes.py -k <selected_test>
"""

from pytest import mark

from nipype import Node
from nipype.interfaces.utility import Select, Function

from narps_open.core import nodes

class TestNodeCreator:
    """ A class that contains all the unit tests for the NodeCreator class."""

    @staticmethod
    @mark.unit_test
    def test_create_node():
        """ Test the create_node method """

        # Define another child for NodeCreator
        class ValidNC(nodes.NodeCreator):
            """ A valid implementation of a NodeCreator """

            @staticmethod
            def create_node(name: str) -> Node:
                """ Return a Node, as expected """
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

        test_node = nodes.RemoveDirectoryNodeCreator.create_node('node_name')
        assert isinstance(test_node, Node)
        assert isinstance(test_node.interface, Function)
        assert test_node.name == 'node_name'

class TestRemoveFileNodeCreator:
    """ A class that contains all the unit tests for the RemoveFileNodeCreator class."""

    @staticmethod
    @mark.unit_test
    def test_create_node():
        """ Test the create_node method """

        test_node = nodes.RemoveFileNodeCreator.create_node('node_name')
        assert isinstance(test_node, Node)
        assert isinstance(test_node.interface, Function)
        assert test_node.name == 'node_name'
