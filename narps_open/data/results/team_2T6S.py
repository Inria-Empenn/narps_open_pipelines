#!/usr/bin/python
# coding: utf-8

""" This module specializes results collection for the 2T6S team """

from os.path import join

from shutil import copyfile

from nibabel import load, save, Nifti1Image

from narps_open.data.results import ResultsCollection

class ResultsCollection2T6S(ResultsCollection):
    """ A class to handle the 2T6S Neurovault collection. """

    def __init__(self):
        super().__init__('2T6S')

    def rectify(self):
        """ Change the signs for values inside unthresholded hypo5 and hypo6,
        as negative values in the images represent negative activation.
        """
        input_files = ['hypo5_unthresh.nii.gz', 'hypo6_unthresh.nii.gz']
        original_files = [
            'hypo5_unthresh_original.nii.gz',
            'hypo6_unthresh_original.nii.gz'
            ]
        rectified_files = ['hypo5_unthresh.nii.gz', 'hypo6_unthresh.nii.gz']

        for input_file, original_file, rectified_file in zip(
            input_files,
            original_files,
            rectified_files
            ):

            input_file = join(self.directory, input_file)
            original_file = join(self.directory, original_file)
            rectified_file = join(self.directory, rectified_file)

            # Make a backup copy of the original file
            copyfile(input_file, original_file)

            # Change the signs of values inside the input image
            input_image = load(input_file)
            save(Nifti1Image(input_image.get_fdata()*-1, input_image.affine), rectified_file)
