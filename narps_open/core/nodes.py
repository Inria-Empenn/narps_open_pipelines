#!/usr/bin/python
# coding: utf-8

""" Generate useful and recurrent nodes to write pipelines """

from abc import ABC, abstractmethod

from nipype import Node
from nipype.interfaces.utility import Function

from narps_open.core.common import remove_directory, remove_file

class NodeCreator(ABC):
    """ An abstract class to shape what node creators must provide """

    @staticmethod
    @abstractmethod
    def create_node(name: str) -> Node:
        """ Return a new Node (the interface of the Node is defined by specialized classes)
            Arguments:
                name, str : the name of the node
        """

class RemoveDirectoryNodeCreator(NodeCreator):
    """ A node creator that provides an interface allowing to remove a directory """

    @staticmethod
    def create_node(name: str) -> Node:
        return Node(Function(
            function = remove_directory,
            input_names = ['_', 'directory_name'],
            output_names = []
            ), name = name)

class RemoveFileNodeCreator(NodeCreator):
    """ A node creator that provides an interface allowing to remove a file """

    @staticmethod
    def create_node(name: str) -> Node:
        return Node(Function(
            function = remove_file,
            input_names = ['_', 'file_name'],
            output_names = []
            ), name = name)
