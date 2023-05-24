#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.utils.correlation' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_correlation.py
    pytest -q test_correlation.py -k <selected_test>
"""

from os import remove
from os.path import exists
from math import isclose

from pytest import raises, fixture, mark
from nibabel import Nifti1Image, save
from numpy import nan, isnan, eye, zeros, full

from narps_open.utils.correlation import (
    mask_using_nan,
    mask_using_zeros,
    get_correlation_coefficient
    )

@fixture
def remove_temporary_files():
    """ A fixture to remove temporary files created by tests """
    yield # test runs here

    for file in [
        'tmp_image_1.nii',
        'tmp_image_2.nii'
    ]:
        if exists(file):
            remove(file)

class TestUtilsCorrelation:
    """ A class that contains all the unit tests for the correlation module."""

    @staticmethod
    @mark.unit_test
    def test_mask_nan():
        """ Test the mask_using_nan function """
        # 1 - Create an image
        dimensions = [2, 2, 2]
        data_test = zeros(dimensions)
        data_test[0, 0, 0] = 1.0
        data_test[0, 1, 0] = 2.0
        data_test[1, 0, 1] = 3.0
        data_test[1, 1, 0] = 4.0
        test_image = Nifti1Image(data_test, affine = eye(4))

        # 2 - Test masking the image
        out_image = mask_using_nan(test_image)
        out_image_data = out_image.get_fdata()
        assert isclose(out_image_data[0, 0, 0], 1.0)
        assert isnan(out_image_data[0, 0, 1])
        assert isclose(out_image_data[0, 1, 0], 2.0)
        assert isnan(out_image_data[0, 1, 1])
        assert isclose(out_image_data[1, 0, 1], 3.0)
        assert isnan(out_image_data[1, 0, 0])
        assert isclose(out_image_data[1, 1, 0], 4.0)
        assert isnan(out_image_data[1, 1, 1])

    @staticmethod
    @mark.unit_test
    def test_mask_zeros():
        """ Test the mask_using_zeros function """
        # 1 - Create an image
        dimensions = [2, 2, 2]
        data_test = full(dimensions, nan)
        data_test[0, 0, 0] = 1.0
        data_test[0, 1, 0] = 2.0
        data_test[1, 0, 1] = 3.0
        data_test[1, 1, 0] = 4.0
        test_image = Nifti1Image(data_test, affine = eye(4))

        # 2 - Test masking the image
        out_image = mask_using_zeros(test_image)
        out_image_data = out_image.get_fdata()
        assert isclose(out_image_data[0, 0, 0], 1.0)
        assert isclose(out_image_data[0, 0, 1], 0.0)
        assert isclose(out_image_data[0, 1, 0], 2.0)
        assert isclose(out_image_data[0, 1, 1], 0.0)
        assert isclose(out_image_data[1, 0, 1], 3.0)
        assert isclose(out_image_data[1, 0, 0], 0.0)
        assert isclose(out_image_data[1, 1, 0], 4.0)
        assert isclose(out_image_data[1, 1, 1], 0.0)

    @staticmethod
    @mark.unit_test
    def test_correlation(remove_temporary_files):
        """ Test the get_correlation_coefficient function, normal usecases """
        # 1 - Create an image & save it to the working directory
        dimensions = [2, 2, 2]
        data_test = zeros(dimensions)
        data_test[0, 0, 0] = 1.0
        data_test[0, 1, 0] = 2.0
        data_test[1, 0, 1] = 3.0
        data_test[1, 1, 0] = 4.0
        test_image = Nifti1Image(data_test, affine = eye(4))
        save(test_image, 'tmp_image_1.nii')

        # 2 - Test correlation with identical images
        assert isclose(
            get_correlation_coefficient('tmp_image_1.nii', 'tmp_image_1.nii'), 1.0)
        assert isclose(
            get_correlation_coefficient('tmp_image_1.nii', 'tmp_image_1.nii', 'pearson'), 1.0)
        assert isclose(
            get_correlation_coefficient('tmp_image_1.nii', 'tmp_image_1.nii', 'spearman'), 1.0)

        # 3 - Create another image with only one value in common with the first one
        dimensions = [2, 2, 2]
        data_test = zeros(dimensions)
        data_test[0, 0, 0] = 1.0
        data_test[0, 1, 0] = 0.0
        data_test[1, 0, 1] = 0.0
        data_test[1, 1, 0] = 0.0
        test_image_2 = Nifti1Image(data_test, affine = eye(4))
        save(test_image_2, 'tmp_image_2.nii')

        # 4 - Test correlation with the images
        assert isclose(
            get_correlation_coefficient('tmp_image_1.nii', 'tmp_image_2.nii'),
            -0.06388765649999398)
        assert isclose(
            get_correlation_coefficient('tmp_image_1.nii', 'tmp_image_2.nii', 'spearman'),
            0.08787495503274935)

    @staticmethod
    @mark.unit_test
    def test_correlation_wrong(remove_temporary_files):
        """ Test the get_correlation_coefficient function, error cases """
        # 1 - Create an image & save it to the working directory
        dimensions = [2, 2, 2]
        data_test = zeros(dimensions)
        data_test[0, 0, 0] = 1.0
        data_test[0, 1, 0] = 2.0
        data_test[1, 0, 1] = 3.0
        data_test[1, 1, 0] = 4.0
        test_image = Nifti1Image(data_test, affine = eye(4))
        save(test_image, 'tmp_image_1.nii')

        # 2 - Use unknown method
        with raises(AttributeError):
            get_correlation_coefficient('tmp_image_1.nii', 'tmp_image_1.nii', 'wrong_method')
