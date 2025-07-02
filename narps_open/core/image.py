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

def get_image_timepoint(in_file: str, time_point: int):
    """
    Extract the 3D volume at a given time point from a 4D Nifti image.
    Create a Nifti file containing the 3D volume.
    Return the filename of the created 3D ouput file.

    Arguments:
        in_file: str, string that represent an absolute path to a Nifti 4D image.
        time_point: int, zero-based index of the volume to extract from the 4D image.

    Returns:
        str, path to the created file
    """
    from os import extsep
    from os.path import abspath, basename
    import nibabel as nib

    out_file = abspath(basename(in_file).split(extsep)[0]+f'_timepoint-{time_point}.nii')
    nib.save(nib.load(in_file).slicer[..., time_point], out_file)

    return out_file
