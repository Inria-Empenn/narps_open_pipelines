import os

from os import path
from os.path import join as opj
import pandas as pd
from typing import List


def participants_tsv():

    participants_tsv = opj(directories()["exp"], "participants.tsv")

    return pd.read_csv(participants_tsv, sep="\t")


def load_config():

    config_file = opj(
        path.dirname(path.realpath(__file__)), "analysis_pipelines_config.tsv"
    )

    return pd.read_csv(config_file, sep="\t")


def return_team_config(team_ID: str) -> dict:
    """
    Args:
        team_ID (str):

    Returns:
        dict: configuration for analysis for this team

    Partly read from ``src/lib/analysis_pipelines_config.tsv``

    keys:

    - ``teamID``
    - ``n_participants``
    - ``excluded_participants``
    - ``func_fwhm``
    - ``directories`` see the ``directories`` function
    - ``subject_list``: List[str] of the subject labels after filtering
      the ``excluded_participants``

    """

    config = load_config()

    config = config[config["teamID"] == team_ID]

    if not config.empty:
        config = config.to_dict("records")[0]
        config["excluded_participants"] = config["excluded_participants"].split(", ")
    else:
        config = {
            "teamID": None,
            "n_participants": 108,
            "excluded_participants": None,
            "func_fwhm": None,
        }

    config["subject_list"] = return_subjects_list(
        team_ID, config["excluded_participants"]
    )

    config["directories"] = directories(team_ID)

    # ensure that we have the number of subjects used by the original team
    assert len(config["subject_list"]) == config["n_participants"]

    return config


def return_subjects_list(team_ID=None, excluded_participants=None) -> List[str]:
    """

    Args:
        team_ID (str, optional): Defaults to None.

    Returns:
        List[str]: subjects labels list
    """

    tmp = directories(team_ID)

    dir_list = os.listdir(tmp["exp"])

    if team_ID is None or excluded_participants is None:
        return [dirs[-3:] for dirs in dir_list if dirs[:3] == "sub"]
    else:
        return [
            dirs[-3:]
            for dirs in dir_list
            if dirs[:3] == "sub" and dirs[4:] not in excluded_participants
        ]


def directories(team_ID: str) -> dict:
    """
    Args:
        team_ID (str)

    Returns:
        dict: dictionary of directories

    keys:

    - ``exp``: directory where the ds001734-download repository is stored
    - ``resulr``: directory where the intermediate and final repositories will be stored
    - ``working``: name of the directory where intermediate results will be stored
    - ``output``: name of the directory where final results will be stored

    """

    if team_ID is None:
        team_ID = "None"

    root_dir = path.abspath(opj(path.dirname(path.realpath(__file__)), "..", ".."))

    # exp_dir : where the data are stored (where the ds001734 directory is stored)
    exp_dir = opj(root_dir, "data", "original", "ds001734")

    # result_dir : where the intermediate and final results will be store
    result_dir = opj(root_dir, "data", "derived", "reproduced")

    # working_dir : where the intermediate outputs will be store
    working_dir = f"NARPS-{team_ID}-reproduced/intermediate_results"

    # output_dir : where the final results will be store
    output_dir = f"NARPS-{team_ID}-reproduced"

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

    anat_file = opj("sub-{subject_id}", "anat", "sub-{subject_id}_T1w.nii.gz")

    func_file = opj(
        "sub-{subject_id}",
        "func",
        "sub-{subject_id}_task-" + task + "_run-{run_id}_bold.nii.gz",
    )

    event_file = opj(
        "sub-{subject_id}",
        "func",
        "sub-{subject_id}_task-" + task + "_run-*_events.tsv",
    )

    magnitude_file = opj(
        "sub-{subject_id}", "fmap", "sub-{subject_id}_magnitude1.nii.gz"
    )

    phasediff_file = opj(
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

    func_preproc = opj(
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

    confounds_file = opj(
        "derivatives",
        "fmriprep",
        "sub-{subject_id}",
        "func",
        "sub-{subject_id}_task-" + task + "_run-{run_id}_bold_confounds.tsv",
    )

    return {"func_preproc": func_preproc, "confounds_file": confounds_file}
