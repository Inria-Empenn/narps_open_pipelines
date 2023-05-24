#!/usr/bin/python
# coding: utf-8

""" A set of utils functions for the narps_open package """

from os.path import join, abspath, dirname, realpath

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
