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
from nipype import Node, Function, Workflow

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

    @staticmethod
    @mark.unit_test
    def test_node_elements_in_string():
        """ Test the elements_in_string function as a nipype.Node """

        # Inputs
        string = 'test_string'
        elements_false = ['z', 'u', 'warning']
        elements_true = ['z', 'u', 'warning', '_']

        # Create a Nipype Node using elements_in_string
        test_node = Node(Function(
            function = co.elements_in_string,
            input_names = ['input_str', 'elements'],
            output_names = ['output']
            ), name = 'test_node')
        test_node.inputs.input_str = string
        test_node.inputs.elements = elements_true
        out = test_node.run().outputs.output

        # Check return value
        assert out == string

        # Change input and check return value
        test_node = Node(Function(
            function = co.elements_in_string,
            input_names = ['input_str', 'elements'],
            output_names = ['output']
            ), name = 'test_node')
        test_node.inputs.input_str = string
        test_node.inputs.elements = elements_false
        out = test_node.run().outputs.output
        assert out is None

    @staticmethod
    @mark.unit_test
    def test_connect_elements_in_string(remove_test_dir):
        """ Test the elements_in_string function as evaluated in a connect """

        # Inputs
        string = 'test_string'
        elements_false = ['z', 'u', 'warning']
        elements_true = ['z', 'u', 'warning', '_']
        function = lambda in_value: in_value

        # Create Nodes
        node_1 = Node(Function(
            function = function,
            input_names = ['in_value'],
            output_names = ['out_value']
            ), name = 'node_1')
        node_1.inputs.in_value = string
        node_true = Node(Function(
            function = function,
            input_names = ['in_value'],
            output_names = ['out_value']
            ), name = 'node_true')
        node_false = Node(Function(
            function = function,
            input_names = ['in_value'],
            output_names = ['out_value']
            ), name = 'node_false')

        # Create Workflow
        test_workflow = Workflow(
            base_dir = TEMPORARY_DIR,
            name = 'test_workflow'
            )
        test_workflow.connect([
            # elements_in_string is evaluated as part of the connection
            (node_1, node_true, [(
                ('out_value', co.elements_in_string, elements_true), 'in_value')]),
            (node_1, node_false, [(
                ('out_value', co.elements_in_string, elements_false), 'in_value')])
            ])

        test_workflow.run()

        test_file_t = join(TEMPORARY_DIR, 'test_workflow', 'node_true', '_report', 'report.rst')
        with open(test_file_t, 'r', encoding = 'utf-8') as file:
            assert '* out_value : test_string' in file.read()

        test_file_f = join(TEMPORARY_DIR, 'test_workflow', 'node_false', '_report', 'report.rst')
        with open(test_file_f, 'r', encoding = 'utf-8') as file:
            assert '* out_value : None' in file.read()

    @staticmethod
    @mark.unit_test
    def test_node_clean_list():
        """ Test the clean_list function as a nipype.Node """

        # Inputs
        input_list = ['z', '_', 'u', 'warning', '_', None]
        element_to_remove_1 = '_'
        output_list_1 = ['z', 'u', 'warning', None]
        element_to_remove_2 = None
        output_list_2 = ['z', '_', 'u', 'warning', '_']

        # Create a Nipype Node using clean_list
        test_node = Node(Function(
            function = co.clean_list,
            input_names = ['input_list', 'element'],
            output_names = ['output']
            ), name = 'test_node')
        test_node.inputs.input_list = input_list
        test_node.inputs.element = element_to_remove_1

        # Check return value
        assert test_node.run().outputs.output == output_list_1

        # Change input and check return value
        test_node = Node(Function(
            function = co.clean_list,
            input_names = ['input_list', 'element'],
            output_names = ['output']
            ), name = 'test_node')
        test_node.inputs.input_list = input_list
        test_node.inputs.element = element_to_remove_2

        assert test_node.run().outputs.output == output_list_2

    @staticmethod
    @mark.unit_test
    def test_connect_clean_list(remove_test_dir):
        """ Test the clean_list function as evaluated in a connect """

        # Inputs
        input_list = ['z', '_', 'u', 'warning', '_', None]
        element_to_remove_1 = '_'
        output_list_1 = ['z', 'u', 'warning', None]
        element_to_remove_2 = None
        output_list_2 = ['z', '_', 'u', 'warning', '_']
        function = lambda in_value: in_value

        # Create Nodes
        node_0 = Node(Function(
            function = function,
            input_names = ['in_value'],
            output_names = ['out_value']
            ), name = 'node_0')
        node_0.inputs.in_value = input_list
        node_1 = Node(Function(
            function = function,
            input_names = ['in_value'],
            output_names = ['out_value']
            ), name = 'node_1')
        node_2 = Node(Function(
            function = function,
            input_names = ['in_value'],
            output_names = ['out_value']
            ), name = 'node_2')

        # Create Workflow
        test_workflow = Workflow(
            base_dir = TEMPORARY_DIR,
            name = 'test_workflow'
            )
        test_workflow.connect([
            # elements_in_string is evaluated as part of the connection
            (node_0, node_1, [(('out_value', co.clean_list, element_to_remove_1), 'in_value')]),
            (node_0, node_2, [(('out_value', co.clean_list, element_to_remove_2), 'in_value')])
            ])
        test_workflow.run()

        test_file_1 = join(TEMPORARY_DIR, 'test_workflow', 'node_1', '_report', 'report.rst')
        with open(test_file_1, 'r', encoding = 'utf-8') as file:
            assert f'* out_value : {output_list_1}' in file.read()

        test_file_2 = join(TEMPORARY_DIR, 'test_workflow', 'node_2', '_report', 'report.rst')
        with open(test_file_2, 'r', encoding = 'utf-8') as file:
            assert f'* out_value : {output_list_2}' in file.read()

    @staticmethod
    @mark.unit_test
    def test_node_list_intersection():
        """ Test the list_intersection function as a nipype.Node """

        # Inputs / outputs
        input_list_1 = ['001', '002', '003', '004']
        input_list_2 = ['002', '004']
        input_list_3 = ['001', '003', '005']
        output_list_1 = ['002', '004']
        output_list_2 = ['001', '003']

        # Create a Nipype Node using list_intersection
        test_node = Node(Function(
            function = co.list_intersection,
            input_names = ['list_1', 'list_2'],
            output_names = ['output']
            ), name = 'test_node')
        test_node.inputs.list_1 = input_list_1
        test_node.inputs.list_2 = input_list_2

        # Check return value
        assert test_node.run().outputs.output == output_list_1

        # Change input and check return value
        test_node = Node(Function(
            function = co.list_intersection,
            input_names = ['list_1', 'list_2'],
            output_names = ['output']
            ), name = 'test_node')
        test_node.inputs.list_1 = input_list_1
        test_node.inputs.list_2 = input_list_3

        assert test_node.run().outputs.output == output_list_2

    @staticmethod
    @mark.unit_test
    def test_connect_list_intersection(remove_test_dir):
        """ Test the list_intersection function as evaluated in a connect """

        # Inputs / outputs
        input_list_1 = ['001', '002', '003', '004']
        input_list_2 = ['002', '004']
        input_list_3 = ['001', '003', '005']
        output_list_1 = ['002', '004']
        output_list_2 = ['001', '003']
        function = lambda in_value: in_value

        # Create Nodes
        node_0 = Node(Function(
            function = function,
            input_names = ['in_value'],
            output_names = ['out_value']
            ), name = 'node_0')
        node_0.inputs.in_value = input_list_1
        node_1 = Node(Function(
            function = function,
            input_names = ['in_value'],
            output_names = ['out_value']
            ), name = 'node_1')
        node_2 = Node(Function(
            function = function,
            input_names = ['in_value'],
            output_names = ['out_value']
            ), name = 'node_2')

        # Create Workflow
        test_workflow = Workflow(
            base_dir = TEMPORARY_DIR,
            name = 'test_workflow'
            )
        test_workflow.connect([
            # elements_in_string is evaluated as part of the connection
            (node_0, node_1, [(('out_value', co.list_intersection, input_list_2), 'in_value')]),
            (node_0, node_2, [(('out_value', co.list_intersection, input_list_3), 'in_value')])
            ])
        test_workflow.run()

        test_file_1 = join(TEMPORARY_DIR, 'test_workflow', 'node_1', '_report', 'report.rst')
        with open(test_file_1, 'r', encoding = 'utf-8') as file:
            assert f'* out_value : {output_list_1}' in file.read()

        test_file_2 = join(TEMPORARY_DIR, 'test_workflow', 'node_2', '_report', 'report.rst')
        with open(test_file_2, 'r', encoding = 'utf-8') as file:
            assert f'* out_value : {output_list_2}' in file.read()
