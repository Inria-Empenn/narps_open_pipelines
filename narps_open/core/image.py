#!/usr/bin/python
# coding: utf-8

""" Image functions to write pipelines """

def get_voxel_dimensions(image: str) -> list:
    """ 
    Return the voxel dimensions of a image in millimeters.

    Arguments:
        image: str, string that represent an absolute path to a Nifti image.

    Returns:
        list, size of the voxels in the image in millimeters.
    """
    # This import must stay inside the function, as required by Nipype
    from nibabel import load

    voxel_dimensions = load(image).header.get_zooms()

    return [
        float(voxel_dimensions[0]),
        float(voxel_dimensions[1]),
        float(voxel_dimensions[2])
        ]
