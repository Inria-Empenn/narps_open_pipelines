#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.core.image' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_image.py
    pytest -q test_image.py -k <selected_test>
"""

from os.path import abspath, join, basename, exists
from filecmp import cmp
from shutil import copyfile

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

    @staticmethod
    @mark.unit_test
    def test_get_image_timepoints(temporary_data_dir):
        """ Test the get_image_timepoints function """

        # Path to the test image
        test_file_path = abspath(join(
            Configuration()['directories']['test_data'],
            'core',
            'image',
            'test_timepoint_input.nii.gz'))

        # Create a Nipype Node using get_image_timepoint
        test_get_image_timepoint = Node(Function(
            function = im.get_image_timepoints,
            input_names = ['in_file', 'start_time_point', 'end_time_point'],
            output_names = ['out_file']
            ), name = 'test_get_image_timepoint')
        test_get_image_timepoint.inputs.in_file = test_file_path
        test_get_image_timepoint.inputs.start_time_point = 10
        test_get_image_timepoint.inputs.end_time_point = 10
        test_get_image_timepoint.base_dir = temporary_data_dir
        outputs = test_get_image_timepoint.run().outputs

        # Check output file name
        assert basename(outputs.out_file) == 'test_timepoint_input_timepoint-10-10.nii'

        # Check if file exists
        assert exists(outputs.out_file)

        # Compare with expected
        out_test_file_path = abspath(join(
            Configuration()['directories']['test_data'],
            'core',
            'image',
            'test_timepoint_output.nii'))
        assert cmp(outputs.out_file, out_test_file_path)

        # Try a new time range
        test_get_image_timepoint = Node(Function(
            function = im.get_image_timepoints,
            input_names = ['in_file', 'start_time_point', 'end_time_point'],
            output_names = ['out_file']
            ), name = 'test_get_image_timepoint')
        test_get_image_timepoint.inputs.in_file = test_file_path
        test_get_image_timepoint.inputs.start_time_point = 2
        test_get_image_timepoint.inputs.end_time_point = 4
        test_get_image_timepoint.base_dir = temporary_data_dir
        outputs = test_get_image_timepoint.run().outputs

        # Check output file name
        assert basename(outputs.out_file) == 'test_timepoint_input_timepoint-2-4.nii'

        # Check if file exists
        assert exists(outputs.out_file)

        # Compare with expected
        out_test_file_path = abspath(join(
            Configuration()['directories']['test_data'],
            'core',
            'image',
            'test_timepoint_output_2.nii'))
        assert cmp(outputs.out_file, out_test_file_path)
