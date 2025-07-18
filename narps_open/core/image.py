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

def get_image_timepoints(in_file: str, start_time_point: int, end_time_point: int) -> str:
    """
    Extract the 3D volume at a given time point from a 4D Nifti image.
    Create a Nifti file containing the 3D volume.
    Return the filename of the created 3D output file.

    Arguments:
        in_file: str, string that represent an absolute path to a Nifti 4D image.
        start_time_point: int, zero-based index of the first volume to extract from the 4D image.
        end_time_point: int, zero-based index of the last volume to extract from the 4D image.

    Returns:
        str, path to the created file
    """
    # These imports must stay inside the function, as required by Nipype
    from os import extsep
    from os.path import abspath, basename
    import nibabel as nib

    # The output filename is based on the base filename of the input file.
    # Note that we use abspath to write the file in the base_directory of a nipype Node.
    out_file = abspath(
        basename(in_file).split(extsep)[0]+f'_timepoint-{start_time_point}-{end_time_point}.nii')

    # Perform timepoints extraction
    nib.save(nib.load(in_file).slicer[..., start_time_point:end_time_point+1], out_file)

    return out_file
