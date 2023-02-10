#!/usr/bin/python
# coding: utf-8

""" Utils functions to perform correlation analyses """

from numpy import corrcoef, reshape, nan, isnan
from scipy.stats import spearmanr
from nibabel import load, Nifti1Image
from nibabel.processing import resample_from_to

def mask_using_nan(data_image: Nifti1Image) -> Nifti1Image:
    """ Mask an image by replacing zeros with NaNs.

        Arguments:
            - data_image, nibabel.Nifti1Image : the image to mask

        Returns:
            - the masked image as nibabel.Nifti1Image
    """

    # Get data from the image
    data = data_image.get_fdata()

    # Replace zeros by NaNs
    data[data == 0.0] = nan

    # Return data as an image
    return Nifti1Image(data, data_image.affine)

def mask_using_zeros(data_image: Nifti1Image) -> Nifti1Image:
    """ Mask an image by replacing NaNs with zeros.

        Arguments:
            - data_image, nibabel.Nifti1Image : the image to mask

        Returns:
            - the masked image as nibabel.Nifti1Image
    """

    # Get data from the image
    data = data_image.get_fdata()

    # Replace NaNs by zeros
    data[isnan(data)] = 0.0

    # Return data as an image
    return Nifti1Image(data, data_image.affine)

def get_correlation_coefficient(
    file_1: str, file_2: str, method: str = 'pearson') -> float:
    """ Return the correlation coefficient of two images.

        Arguments :
            - file_1, str - path to the first image
            - file_2, str - path to the second image ; file_2 will be resampled on file_1
            - method, str - either 'pearson', or 'spearman': the correlation method to use
            - reslice_on_file_2, bool - set to :
                - True if you wish to reslice file_1 on file_2
                - False otherwise

        Returns :
            - _, float - the correlation coefficient of the two input images,
            using the passed method
    """

    # Load images
    image_1 = load(file_1)
    image_2 = load(file_2)

    # Set masking using NaN's
    image_1 = mask_using_zeros(image_1)
    image_2 = mask_using_zeros(image_2)

    # Resample using nearest nneighbours
    image_2 = resample_from_to(image_2, image_1, order = 0)

    # Make 1D vectors from the images data
    data_1 = reshape(image_1.get_fdata(), -1)
    data_2 = reshape(image_2.get_fdata(), -1)

    # Compute the correlation coefficient
    if method == 'pearson':
        return corrcoef(data_1, data_2)[0][1]
    if method == 'spearman':
        return spearmanr(data_1, data_2).correlation

    raise AttributeError(f'Wrong correlation method provided: {method}.')
