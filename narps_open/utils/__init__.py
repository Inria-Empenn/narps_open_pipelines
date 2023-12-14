#!/usr/bin/python
# coding: utf-8

""" A set of utils functions for the narps_open package """

from os import listdir
from os.path import isfile, join, abspath, dirname, realpath, splitext

from nibabel import load
from hashlib import sha256

def show_download_progress(count, block_size, total_size):
    """ A hook function to be passed to urllib.request.urlretrieve in order to
        print the progress of a download.

        Arguments:
        - count: int - the number of blocks already downloaded
        - block_size: int - the size in bytes of a block
        - total_size: int - the total size in bytes of the download. -1 if not provided.
    """
    if total_size != -1:
        # Display a percentage
        display_value = str(int(count * block_size * 100 / total_size))+' %'
    else:
        # Draw a pretty cursor
        cursor = ['⣾','⣽','⣻','⢿','⡿','⣟','⣯','⣷']
        display_value = cursor[int(count)%len(cursor)]

    # Showing download progress
    print('Downloading', display_value, end='\r')

def hash_image(path_img: str) -> str:
    """ Return the sha256 hash of a nifti image
        Arguments:
        - path_img, str: path to the nifti image
    """

    # Load image
    image = load(path_img)

    # Hash data
    hasher = sha256()
    for element in image.affine.ravel():
        hasher.update(element)
    for element in image.header:
        hasher.update(element.encode(encoding='utf-8'))
    for element in image.get_fdata().ravel():
        hasher.update(element)

    return hasher.hexdigest()

def hash_dir_images(path: str) -> str:
    """ Return the sha256 hash of hashes of nifti images inside a directory
        Arguments:
        - path, str: path to the directory
    """
    # Create the list of images
    image_list = []
    for file in listdir(path):
        if isfile(join(path, file)) and file.endswith('.nii.gz'):
            image_list.append(join(path, file))

    # Hash data
    hasher = sha256()
    for image in sorted(image_list):
        hasher.update(hash_image(image).encode(encoding = 'utf-8'))

    return hasher.hexdigest()

def get_subject_id(file_name: str) -> str:
    """ Return the id of the subject corresponding to the passed file name.
        Return None if the file name is not associated with any subject.
        TODO : a feature to be handled globally to parse data in a file name.
    """
    key = 'subject_id'
    if key not in file_name:
        return None

    position = file_name.find(key) + len(key) + 1

    return file_name[position:position+3]

def directories(team_id: str) -> dict:
    """
    Args:
        team_id (str)

    Returns:
        dict: dictionary of directories

    keys:

    - ``exp``: directory where the ds001734-download repository is stored
    - ``resulr``: directory where the intermediate and final repositories will be stored
    - ``working``: name of the directory where intermediate results will be stored
    - ``output``: name of the directory where final results will be stored

    """

    if team_id is None:
        team_id = "None"

    root_dir = abspath(join(dirname(realpath(__file__)), "..", ".."))

    # exp_dir : where the data are stored (where the ds001734 directory is stored)
    exp_dir = join(root_dir, "data", "original", "ds001734")

    # result_dir : where the intermediate and final results will be store
    result_dir = join(root_dir, "data", "derived", "reproduced")

    # working_dir : where the intermediate outputs will be store
    working_dir = f"NARPS-{team_id}-reproduced/intermediate_results"

    # output_dir : where the final results will be store
    output_dir = f"NARPS-{team_id}-reproduced"

    return {
        "root": root_dir,
        "exp": exp_dir,
        "output": output_dir,
        "working": working_dir,
        "result": result_dir,
    }

def raw_data_template() -> dict:
    """
    Returns:
        dict: dictionary of filename templates for the raw dataset
        to pass to the SelectFiles nipype interface
    """

    task = "MGT"

    anat_file = join("sub-{subject_id}", "anat", "sub-{subject_id}_T1w.nii.gz")

    func_file = join(
        "sub-{subject_id}",
        "func",
        "sub-{subject_id}_task-" + task + "_run-{run_id}_bold.nii.gz",
    )

    event_file = join(
        "sub-{subject_id}",
        "func",
        "sub-{subject_id}_task-" + task + "_run-*_events.tsv",
    )

    magnitude_file = join(
        "sub-{subject_id}", "fmap", "sub-{subject_id}_magnitude1.nii.gz"
    )

    phasediff_file = join(
        "sub-{subject_id}", "fmap", "sub-{subject_id}_phasediff.nii.gz"
    )

    return {
        "anat": anat_file,
        "func": func_file,
        "event_file": event_file,
        "magnitude": magnitude_file,
        "phasediff": phasediff_file,
    }

def fmriprep_data_template() -> dict:
    """
    Returns:
        dict: dictionary of filename templates for the fmriprep dataset
        to pass to the SelectFiles nipype interface
    """

    task = "MGT"
    space = "MNI152NLin2009cAsym"

    func_preproc = join(
        "derivatives",
        "fmriprep",
        "sub-{subject_id}",
        "func",
        "sub-{subject_id}_"
        + task
        + "-MGT_run-{run_id}_bold_space-"
        + space
        + "_preproc.nii.gz",
    )

    confounds_file = join(
        "derivatives",
        "fmriprep",
        "sub-{subject_id}",
        "func",
        "sub-{subject_id}_task-" + task + "_run-{run_id}_bold_confounds.tsv",
    )

    return {"func_preproc": func_preproc, "confounds_file": confounds_file}
