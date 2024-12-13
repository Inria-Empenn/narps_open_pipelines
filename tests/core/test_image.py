#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.core.image' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_image.py
    pytest -q test_image.py -k <selected_test>
"""

from os.path import abspath, join
from numpy import isclose

from pytest import mark
from nipype import Node, Function

from narps_open.utils.configuration import Configuration
import narps_open.core.image as im

class TestCoreImage:
    """ A class that contains all the unit tests for the image module."""

    @staticmethod
    @mark.unit_test
    def test_get_voxel_dimensions():
        """ Test the get_voxel_dimensions function """

        # Path to the test image
        test_file_path = abspath(join(
            Configuration()['directories']['test_data'],
            'core',
            'image',
            'test_image.nii.gz'))

        # Create a Nipype Node using get_voxel_dimensions
        test_get_voxel_dimensions_node = Node(Function(
            function = im.get_voxel_dimensions,
            input_names = ['image'],
            output_names = ['voxel_dimensions']
            ), name = 'test_get_voxel_dimensions_node')
        test_get_voxel_dimensions_node.inputs.image = test_file_path
        outputs = test_get_voxel_dimensions_node.run().outputs

        # Check voxel sizes
        assert isclose(outputs.voxel_dimensions, [8.0, 8.0, 9.6]).all()
