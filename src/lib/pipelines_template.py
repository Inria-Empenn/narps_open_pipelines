# THIS IS A TEMPLATE THAT CAN BE USE TO REPRODUCE A NEW PIPELINE

from nipype.interfaces.spm import node_function  # Complete with necessary functions
from nipype.interfaces.fsl import node_function  # Complete with necessary functions
from nipype.algorithms.modelgen import (
    SpecifySPMModel,
    SpecifyModel,
)  # Functions used during L1 analysis
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.algorithms.misc import Gunzip
from nipype import Workflow, Node, MapNode, JoinNode
from nipype.interfaces.base import Bunch

from os.path import join as opj
import os
import json

from typing import List


def get_preprocessing(
    exp_dir: str,
    result_dir: str,
    working_dir: str,
    output_dir: str,
    subject_list: List[str],
    run_list: List[str],
    fwhm: float,
):
    """
    Returns the preprocessing workflow.

    Parameters:
        - exp_dir: directory where raw data are stored
        - result_dir: directory where results will be stored
        - working_dir: name of the sub-directory for intermediate results
        - output_dir: name of the sub-directory for final results
        - subject_list: list of subject for which you want to do the preprocessing
        - run_list: list of runs for which you want to do the preprocessing
        - fwhm: fwhm for smoothing step

    Returns:
        - preprocessing: Nipype WorkFlow
    """

    # THE FOLLOWING PART STAYS THE SAME FOR ALL PREPROCESSING PIPELINES
    infosource_preproc = Node(
        IdentityInterface(fields=["subject_id", "run_id"]), name="infosource_preproc"
    )

    infosource_preproc.iterables = [
        ("subject_id", subject_list),
        ("run_id", run_list),
    ]  # Iterates over subject and runs

    # Templates to select files node
    anat_file = opj("sub-{subject_id}", "anat", "sub-{subject_id}_T1w.nii.gz")

    func_file = opj(
        "sub-{subject_id}", "func", "sub-{subject_id}_task-MGT_run-{run_id}_bold.nii.gz"
    )

    magnitude_file = opj(
        "sub-{subject_id}", "fmap", "sub-{subject_id}_magnitude1.nii.gz"
    )

    phasediff_file = opj(
        "sub-{subject_id}", "fmap", "sub-{subject_id}_phasediff.nii.gz"
    )

    template = {
        "anat": anat_file,
        "func": func_file,
        "magnitude": magnitude_file,
        "phasediff": phasediff_file,
    }

    # SelectFiles node - to select necessary files
    selectfiles_preproc = Node(
        SelectFiles(template, base_directory=exp_dir), name="selectfiles_preproc"
    )

    # DataSink Node - store the wanted results in the wanted repository
    datasink_preproc = Node(
        DataSink(base_directory=result_dir, container=output_dir),
        name="datasink_preproc",
    )

    # THE FOLLOWING PART HAS TO BE MODIFIED WITH NODES OF THE PIPELINE
    node_variable = Node(
        node_function, name="node_name"
    )  # Replace with the name of the node_variable,
    # the node_function to use in the NiPype interface,
    # and the name of the node (recommanded to be the same as node_variable)

    # ADD OTHER NODES WITH THE DIFFERENT STEPS OF THE PIPELINE

    preprocessing = Workflow(
        base_dir=opj(result_dir, working_dir), name="preprocessing"
    )  # Workflow creation

    preprocessing.connect(
        [
            (
                infosource_preproc,
                selectfiles_preproc,
                [("subject_id", "subject_id"), ("run_id", "run_id")],
            ),
            (
                selectfiles_preproc,
                node_variable[("func", "node_input_name")],
            ),  # Complete with name of node to link with and the name of the input
            # Input and output names can be found on NiPype documentation
            (
                node_variable,
                datasink_preproc,
                [("node_output_name", "preprocess.@sym_link")],
            ),
        ]
    )

    return preprocessing


# THIS FUNCTION IS USED IN THE FIRST LEVEL ANALYSIS PIPELINES OF SPM
# THIS IS AN EXAMPLE THAT IS ADAPTED TO A SPECIFIC PIPELINE
# MODIFY ACCORDING TO THE PIPELINE YOU WANT TO REPRODUCE
def get_subject_infos_spm(event_files: List[str], runs: List[str]):
    """

    The model contained 6 regressors per run:

    - One predictor with onset at the start of the trial and duration of 4s.

    - Two parametric modulators (one for gains, one for losses)
      were added to the trial onset predictor.
      The two parametric modulators were orthogonalized w.r.t. the main predictor,
      but were not orthogonalized w.r.t. one another.

    - Two predictors modelling the decision output,
      one for accepting the gamble and one for rejecting it
      (merging strong and weak decisions).

    The onset was defined as the beginning of the trial + RT
    and the duration was set to 0 (stick function).
    - One constant term for each run was included (SPM12 default design).

    Create Bunchs for specifySPMModel.

    Parameters :
    - event_files: list of events files (one per run) for the subject
    - runs: list of runs to use

    Returns :
    - subject_info : list of Bunch for 1st level analysis.
    """
    from nipype.interfaces.base import Bunch

    cond_names = ["trial", "accepting", "rejecting"]
    onset = {}
    duration = {}
    weights_gain = {}
    weights_loss = {}
    onset_button = {}
    duration_button = {}

    for r in range(len(runs)):  # Loop over number of runs.
        onset.update(
            {s + "_run" + str(r + 1): [] for s in cond_names}
        )  # creates dictionary items with empty lists
        duration.update({s + "_run" + str(r + 1): [] for s in cond_names})
        weights_gain.update({"gain_run" + str(r + 1): []})
        weights_loss.update({"loss_run" + str(r + 1): []})

    for r, run in enumerate(runs):

        f_events = event_files[r]

        with open(f_events, "rt") as f:
            next(f)  # skip the header

            for line in f:
                info = line.strip().split()

                for cond in cond_names:
                    val = cond + "_run" + str(r + 1)  # trial_run1 or accepting_run1
                    val_gain = "gain_run" + str(r + 1)  # gain_run1
                    val_loss = "loss_run" + str(r + 1)  # loss_run1
                    if cond == "trial":
                        onset[val].append(float(info[0]))  # onsets for trial_run1
                        duration[val].append(float(4))
                        weights_gain[val_gain].append(
                            float(info[2])
                        )  # weights gain for trial_run1
                        weights_loss[val_loss].append(
                            float(info[3])
                        )  # weights loss for trial_run1
                    elif cond == "accepting" and "accept" in info[5]:
                        onset[val].append(float(info[0]) + float(info[4]))
                        duration[val].append(float(0))
                    elif cond == "rejecting" and "reject" in info[5]:
                        onset[val].append(float(info[0]) + float(info[4]))
                        duration[val].append(float(0))

    # Bunching is done per run, i.e. trial_run1, trial_run2, etc.
    # But names must not have '_run1' etc because we concatenate runs
    subject_info = []
    for r in range(len(runs)):

        cond = [s + "_run" + str(r + 1) for s in cond_names]
        gain = "gain_run" + str(r + 1)
        loss = "loss_run" + str(r + 1)

        subject_info.insert(
            r,
            Bunch(
                conditions=cond_names,
                onsets=[onset[c] for c in cond],
                durations=[duration[c] for c in cond],
                amplitudes=None,
                tmod=None,
                pmod=[
                    Bunch(
                        name=["gain", "loss"],
                        poly=[1, 1],
                        param=[weights_gain[gain], weights_loss[loss]],
                    ),
                    None,
                ],
                regressor_names=None,
                regressors=None,
            ),
        )

    return subject_info


# THIS FUNCTION IS USED IN THE FIRST LEVEL ANALYSIS PIPELINES OF FSL
# THIS IS AN EXAMPLE THAT IS ADAPTED TO A SPECIFIC PIPELINE
# MODIFY ACCORDING TO THE PIPELINE YOU WANT TO REPRODUCE
def get_session_infos_fsl(event_file: str):
    """
    Create Bunchs for specifyModel.

    Parameters :
    - event_file : file corresponding to the run and the subject to analyze

    Returns :
    - subject_info : list of Bunch for 1st level analysis.
    """
    from os.path import join as opj
    from nipype.interfaces.base import Bunch
    import numpy as np

    cond_names = ["trial", "gain", "loss"]

    onset = {}
    duration = {}
    amplitude = {}

    for c in cond_names:  # For each condition.
        onset.update({c: []})  # creates dictionary items with empty lists
        duration.update({c: []})
        amplitude.update({c: []})

    with open(event_file, "rt") as f:
        next(f)  # skip the header

        for line in f:
            info = line.strip().split()
            # Creates list with onsets, duration and loss/gain for amplitude (FSL)
            for c in cond_names:
                if c == "gain":
                    onset[c].append(float(info[0]))
                    duration[c].append(float(info[4]))
                    amplitude[c].append(float(info[2]))
                elif c == "loss":
                    onset[c].append(float(info[0]))
                    duration[c].append(float(info[4]))
                    amplitude[c].append(float(info[3]))
                elif c == "trial":
                    onset[c].append(float(info[0]))
                    duration[c].append(float(info[4]))
                    amplitude[c].append(float(1))

    subject_info = []

    subject_info.append(
        Bunch(
            conditions=cond_names,
            onsets=[onset[k] for k in cond_names],
            durations=[duration[k] for k in cond_names],
            amplitudes=[amplitude[k] for k in cond_names],
            regressor_names=None,
            regressors=None,
        )
    )

    return subject_info


# THIS FUNCTION CREATES THE CONTRASTS THAT WILL BE ANALYZED IN THE FIRST LEVEL ANALYSIS
# IT IS ADAPTED FOR A SPECIFIC PIPELINE AND SHOULD BE MODIFIED DEPENDING ON THE PIPELINE YOU ARE TRYING TO REPRODUCE
def get_contrasts(subject_id: str):
    """
    Create the list of tuples that represents contrasts.
    Each contrast is in the form :
    (Name,Stat,[list of condition names],[weights on those conditions])

    Parameters:
        - subject_id: ID of the subject

    Returns:
        - contrasts: list of tuples, list of contrasts to analyze
    """
    # list of condition names
    conditions = ["trial", "trialxgain^1", "trialxloss^1"]

    # create contrasts
    trial = ("trial", "T", conditions, [1, 0, 0])

    effect_gain = ("effect_of_gain", "T", conditions, [0, 1, 0])

    effect_loss = ("effect_of_loss", "T", conditions, [0, 0, 1])

    # contrast list
    contrasts = [effect_gain, effect_loss]

    return contrasts


# FUNCTION TO CREATE THE WORKFLOW OF A L1 ANALYSIS (SUBJECT LEVEL)
def get_l1_analysis(
    exp_dir: str,
    result_dir: str,
    working_dir: str,
    output_dir: str,
    subject_list: List[str],
    run_list: List[str],
    TR: float,
):
    """
    Returns the first level analysis workflow.

    Parameters:
        - exp_dir: directory where raw data are stored
        - result_dir: directory where results will be stored
        - working_dir: name of the sub-directory for intermediate results
        - output_dir: name of the sub-directory for final results
        - subject_list: list of subject for which you want to do the analysis
        - run_list: list of runs for which you want to do the analysis
        - TR: time repetition used during acquisition


    Returns:
        - l1_analysis : Nipype WorkFlow
    """
    # THE FOLLOWING PART STAYS THE SAME FOR ALL PIPELINES
    # Infosource Node - To iterate on subjects
    infosource = Node(
        IdentityInterface(
            fields=["subject_id", "exp_dir", "result_dir", "working_dir", "run_list"],
            exp_dir=exp_dir,
            result_dir=result_dir,
            working_dir=working_dir,
            run_list=run_list,
        ),
        name="infosource",
    )

    infosource.iterables = [("subject_id", subject_list)]  # ITERATES OVER SUBJECT LIST

    # Templates to select files node
    func_file = opj(
        result_dir,
        output_dir,
        "preprocess",
        "_run_id_*_subject_id_{subject_id}",
        "complete_filename_{subject_id}_complete_filename.nii",
    )  # CHANGE THE NAME OF THE FILE DEPENDING ON THE FILENAMES OF RESULTS OF PREPROCESSING

    event_files = opj(
        exp_dir,
        "sub-{subject_id}",
        "func",
        "sub-{subject_id}_task-MGT_run-*_events.tsv",
    )

    template = {"func": func_file, "event": event_files}

    # SelectFiles node - to select necessary files
    selectfiles = Node(
        SelectFiles(template, base_directory=exp_dir), name="selectfiles"
    )

    # DataSink Node - store the wanted results in the wanted repository
    datasink = Node(
        DataSink(base_directory=result_dir, container=output_dir), name="datasink"
    )

    # THIS IS THE NODE EXECUTING THE get_subject_infos_spm FUNCTION
    # IF YOU'RE DOING AN FSL PIPELINE --> JUST CHANGE THE NAME OF THE FUNCTION TO get_subject_infos_fsl
    # Get Subject Info - get subject specific condition information
    subject_infos = Node(
        Function(
            input_names=["event_files", "runs"],
            output_names=["subject_info"],
            function=get_subject_infos_spm,
        ),
        name="subject_infos",
    )

    subject_infos.inputs.runs = run_list
    # THIS IS THE NODE EXECUTING THE get_contrasts FUNCTION
    # Node contrasts to get contrasts
    contrasts = Node(
        Function(
            function=get_contrasts,
            input_names=["subject_id"],
            output_names=["contrasts"],
        ),
        name="contrasts",
    )

    # THE FOLLOWING PART HAS TO BE MODIFIED WITH NODES OF THE PIPELINE
    node_variable = Node(
        node_function, name="node_name"
    )  # Replace with the name of the node_variable,
    # the node_function to use in the NiPype interface,
    # and the name of the node (recommanded to be the same as node_variable)

    # ADD OTHER NODES WITH THE DIFFERENT STEPS OF THE PIPELINE

    # Create l1 analysis workflow and connect its nodes
    l1_analysis = Workflow(base_dir=opj(result_dir, working_dir), name="l1_analysis")

    l1_analysis.connect(
        [
            (infosource, selectfiles, [("subject_id", "subject_id")]),
            (infosource, contrasts, [("subject_id", "subject_id")]),
            (selectfiles, subject_infos, [("event", "event_files")]),
            (
                selectfiles,
                node_variable[("func", "node_input_name")],
            ),  # Complete with name of node to link with and the name of the input
            # Input and output names can be found on NiPype documentation
            (node_variable, datasink, [("node_output_name", "preprocess.@sym_link")]),
        ]
    )

    return l1_analysis


# THIS FUNCTION RETURNS THE LIST OF IDS AND FILES OF EACH GROUP OF PARTICIPANTS TO DO SEPARATE GROUP LEVEL ANALYSIS AND BETWEEN GROUP ANALYSIS
# THIS FUNCTIONS IS ADAPTED FOR AN SPM PIPELINE.
def get_subset_contrasts_spm(
    file_list, method: str, subject_list: List[str], participants_file: str
):
    """
    Parameters :
    - file_list : original file list selected by selectfiles node
    - subject_list : list of subject IDs that are in the wanted group for the analysis
    - participants_file: file containing participants caracteristics
    - method: one of "equalRange", "equalIndifference" or "groupComp"

    This function return the file list containing only the files belonging to subject in the wanted group.
    """
    equalIndifference_id = []
    equalRange_id = []
    equalIndifference_files = []
    equalRange_files = []

    with open(
        participants_file, "rt"
    ) as f:  # Reading file containing participants IDs and groups
        next(f)  # skip the header

        for line in f:
            info = line.strip().split()

            if (
                info[0][-3:] in subject_list and info[1] == "equalIndifference"
            ):  # Checking for each participant if its ID was selected
                # and separate people depending on their group
                equalIndifference_id.append(info[0][-3:])
            elif info[0][-3:] in subject_list and info[1] == "equalRange":
                equalRange_id.append(info[0][-3:])

    # Checking for each selected file if the corresponding participant was selected
    # and add the file to the list corresponding to its group
    for file in file_list:
        sub_id = file.split("/")
        if sub_id[-2][-3:] in equalIndifference_id:
            equalIndifference_files.append(file)
        elif sub_id[-2][-3:] in equalRange_id:
            equalRange_files.append(file)

    return (
        equalIndifference_id,
        equalRange_id,
        equalIndifference_files,
        equalRange_files,
    )


# THIS FUNCTION IS ADAPTED FOR AN FSL PIPELINE.
def get_subgroups_contrasts_fsl(
    copes, varcopes, subject_list: List[str], participants_file: str
):
    """
    Parameters :
    - copes: original file list selected by selectfiles node
    - varcopes: original file list selected by selectfiles node
    - subject_ids: list of subject IDs that are analyzed
    - participants_file: file containing participants characteristics

    This function return the file list containing only the files
    belonging to subject in the wanted group.
    """

    from os.path import join as opj

    equalRange_id = []
    equalIndifference_id = []

    # Reading file containing participants IDs and groups
    with open(participants_file, "rt") as f:
        next(f)  # skip the header

        for line in f:
            info = line.strip().split()

            # Checking for each participant if its ID was selected
            # and separate people depending on their group
            if info[0][-3:] in subject_list and info[1] == "equalIndifference":
                equalIndifference_id.append(info[0][-3:])
            elif info[0][-3:] in subject_list and info[1] == "equalRange":
                equalRange_id.append(info[0][-3:])

    copes_equalIndifference = []
    copes_equalRange = []
    copes_global = []
    varcopes_equalIndifference = []
    varcopes_equalRange = []
    varcopes_global = []

    # Checking for each selected file if the corresponding participant was selected
    # and add the file to the list corresponding to its group
    for file in copes:
        sub_id = file.split("/")
        if sub_id[-2][-3:] in equalIndifference_id:
            copes_equalIndifference.append(file)
        elif sub_id[-2][-3:] in equalRange_id:
            copes_equalRange.append(file)
        if sub_id[-2][-3:] in subject_list:
            copes_global.append(file)

    for file in varcopes:  # Same thing but for varcopes files
        sub_id = file.split("/")
        if sub_id[-2][-3:] in equalIndifference_id:
            varcopes_equalIndifference.append(file)
        elif sub_id[-2][-3:] in equalRange_id:
            varcopes_equalRange.append(file)
        if sub_id[-2][-3:] in subject_list:
            varcopes_global.append(file)

    return (
        copes_equalIndifference,
        copes_equalRange,
        varcopes_equalIndifference,
        varcopes_equalRange,
        equalIndifference_id,
        equalRange_id,
        copes_global,
        varcopes_global,
    )


# THIS FUNCTION CREATES THE DICTIONARY OF REGRESSORS USED IN FSL NIPYPE PIPELINES
def get_regs(
    equalRange_id: List[str],
    equalIndifference_id: List[str],
    method: str,
    subject_list: List[str],
):
    """
    Create dictionary of regressors for group analysis.

    Parameters:
        - equalRange_id: ids of subjects in equal range group
        - equalIndifference_id: ids of subjects in equal indifference group
        - method: one of "equalRange", "equalIndifference" or "groupComp"
        - subject_list: ids of subject for which to do the analysis

    Returns:
        - regressors: dict, dictionary of regressors used to distinguish groups in FSL group analysis
    """
    # For one sample t-test, creates a dictionary with a list of the size of the number of participants
    if method == "equalRange":
        regressors = dict(group_mean=[1 for i in range(len(equalRange_id))])

    elif method == "equalIndifference":
        regressors = dict(group_mean=[1 for i in range(len(equalIndifference_id))])

    # For two sample t-test, creates 2 lists:
    #  - one for equal range group,
    #  - one for equal indifference group
    # Each list contains n_sub values with 0 and 1 depending on the group of the participant
    # For equalRange_reg list --> participants with a 1 are in the equal range group
    elif method == "groupComp":
        equalRange_reg = [
            1 for i in range(len(equalRange_id) + len(equalIndifference_id))
        ]
        equalIndifference_reg = [
            0 for i in range(len(equalRange_id) + len(equalIndifference_id))
        ]

        for i, sub_id in enumerate(subject_list):
            if sub_id in equalIndifference_id:
                index = i
                equalIndifference_reg[index] = 1
                equalRange_reg[index] = 0

        regressors = dict(
            equalRange=equalRange_reg, equalIndifference=equalIndifference_reg
        )

    return regressors


# FUNCTION TO CREATE THE WORKFLOW OF A L2 ANALYSIS (GROUP LEVEL)
def get_l2_analysis(
    exp_dir: str,
    result_dir: str,
    working_dir: str,
    output_dir: str,
    subject_list: List[str],
    contrast_list: List[str],
    n_sub: int,
    method: str,
):
    """
    Returns the 2nd level of analysis workflow.

    Parameters:
        - exp_dir: directory where raw data are stored
        - result_dir: directory where results will be stored
        - working_dir: name of the sub-directory for intermediate results
        - output_dir: name of the sub-directory for final results
        - subject_list: list of subject for which you want to do the preprocessing
        - contrast_list: list of contrasts to analyze
        - n_sub: number of subjects used to do the analysis
        - method: one of "equalRange", "equalIndifference" or "groupComp"

    Returns:
        - l2_analysis: Nipype WorkFlow
    """
    # THE FOLLOWING PART STAYS THE SAME FOR ALL PREPROCESSING PIPELINES
    # Infosource - a function free node to iterate over the list of subject names
    infosource_groupanalysis = Node(
        IdentityInterface(fields=["contrast_id", "subjects"], subjects=subject_list),
        name="infosource_groupanalysis",
    )

    infosource_groupanalysis.iterables = [("contrast_id", contrast_list)]

    # SelectFiles
    contrast_file = opj(
        result_dir,
        output_dir,
        "l1_analysis",
        "_subject_id_*",
        "complete_filename_{contrast_id}_complete_filename.nii",
    )
    # CHANGE THE NAME OF THE FILE DEPENDING ON THE FILENAMES OF THE RESULTS OF PREPROCESSING (DIFFERENT FOR AN FSL PIPELINE)

    participants_file = opj(exp_dir, "participants.tsv")

    templates = {"contrast": contrast_file, "participants": participants_file}

    selectfiles_groupanalysis = Node(
        SelectFiles(templates, base_directory=result_dir, force_list=True),
        name="selectfiles_groupanalysis",
    )

    # Datasink node : to save important files
    datasink_groupanalysis = Node(
        DataSink(base_directory=result_dir, container=output_dir),
        name="datasink_groupanalysis",
    )

    # IF THIS IS AN SPM PIPELINE:
    # Node to select subset of contrasts
    sub_contrasts = Node(
        Function(
            input_names=["file_list", "method", "subject_list", "participants_file"],
            output_names=[
                "equalIndifference_id",
                "equalRange_id",
                "equalIndifference_files",
                "equalRange_files",
            ],
            function=get_subset_contrasts_spm,
        ),
        name="sub_contrasts",
    )

    sub_contrasts.inputs.method = method

    # IF THIS IS AN FSL PIPELINE:
    subgroups_contrasts = Node(
        Function(
            input_names=["copes", "varcopes", "subject_ids", "participants_file"],
            output_names=[
                "copes_equalIndifference",
                "copes_equalRange",
                "varcopes_equalIndifference",
                "varcopes_equalRange",
                "equalIndifference_id",
                "equalRange_id",
                "copes_global",
                "varcopes_global",
            ],
            function=get_subgroups_contrasts,
        ),
        name="subgroups_contrasts",
    )

    regs = Node(
        Function(
            input_names=[
                "equalRange_id",
                "equalIndifference_id",
                "method",
                "subject_list",
            ],
            output_names=["regressors"],
            function=get_regs,
        ),
        name="regs",
    )
    regs.inputs.method = method
    regs.inputs.subject_list = subject_list

    # THE FOLLOWING PART HAS TO BE MODIFIED WITH NODES OF THE PIPELINE
    node_variable = Node(
        node_function, name="node_name"
    )  # Replace with the name of the node_variable,
    # the node_function to use in the NiPype interface,
    # and the name of the node (recommanded to be the same as node_variable)

    # ADD OTHER NODES WITH THE DIFFERENT STEPS OF THE PIPELINE

    l2_analysis = Workflow(
        base_dir=opj(result_dir, working_dir), name=f"l2_analysis_{method}_nsub_{n_sub}"
    )
    # FOR AN SPM PIPELINE
    l2_analysis.connect(
        [
            (
                infosource_groupanalysis,
                selectfiles_groupanalysis,
                [("contrast_id", "contrast_id")],
            ),
            (infosource_groupanalysis, sub_contrasts, [("subjects", "subject_list")]),
            (
                selectfiles_groupanalysis,
                sub_contrasts,
                [("contrast", "file_list"), ("participants", "participants_file")],
            ),  # Complete with other links between nodes
        ]
    )

    # FOR AN FSL PIPELINE
    l2_analysis.connect(
        [
            (
                infosource_groupanalysis,
                selectfiles_groupanalysis,
                [("contrast_id", "contrast_id")],
            ),
            (
                infosource_groupanalysis,
                subgroups_contrasts,
                [("subject_list", "subject_ids")],
            ),
            (
                selectfiles_groupanalysis,
                subgroups_contrasts,
                [
                    ("cope", "copes"),
                    ("varcope", "varcopes"),
                    ("participants", "participants_file"),
                ],
            ),
            (
                selectfiles_groupanalysis,
                node_variable[("func", "node_input_name")],
            ),  # Complete with name of node to link with and the name of the input
            # Input and output names can be found on NiPype documentation
            (
                node_variable,
                datasink_groupanalysis,
                [("node_output_name", "preprocess.@sym_link")],
            ),
        ]
    )  # Complete with other links between nodes

    if method == "equalRange" or method == "equalIndifference":
        contrasts = [("Group", "T", ["mean"], [1]), ("Group", "T", ["mean"], [-1])]

    elif method == "groupComp":
        contrasts = [
            ("Eq range vs Eq indiff in loss", "T", ["Group_{1}", "Group_{2}"], [1, -1])
        ]

    # ADD OTHER NODES WITH THE DIFFERENT STEPS OF THE PIPELINE

    return l2_analysis


# THIS FUNCTION IS USED TO REORGANIZE FINAL RESULTS OF THE PIPELINE
def reorganize_results(result_dir: str, output_dir: str, n_sub: int, team_ID: str):
    """
    Reorganize the results to analyze them.

    Parameters:
        - result_dir: directory where results will be stored
        - output_dir: name of the sub-directory for final results
        - n_sub: number of subject used for the analysis
        - team_ID: ID of the team to reorganize results

    """
    from os.path import join as opj
    import os
    import shutil
    import gzip

    h1 = opj(
        result_dir,
        output_dir,
        f"l2_analysis_equalIndifference_nsub_{n_sub}",
        "_contrast_id_01",
    )
    h2 = opj(
        result_dir,
        output_dir,
        f"l2_analysis_equalRange_nsub_{n_sub}",
        "_contrast_id_01",
    )
    h3 = opj(
        result_dir,
        output_dir,
        f"l2_analysis_equalIndifference_nsub_{n_sub}",
        "_contrast_id_01",
    )
    h4 = opj(
        result_dir,
        output_dir,
        f"l2_analysis_equalRange_nsub_{n_sub}",
        "_contrast_id_01",
    )
    h5 = opj(
        result_dir,
        output_dir,
        f"l2_analysis_equalIndifference_nsub_{n_sub}",
        "_contrast_id_02",
    )
    h6 = opj(
        result_dir,
        output_dir,
        f"l2_analysis_equalRange_nsub_{n_sub}",
        "_contrast_id_02",
    )
    h7 = opj(
        result_dir,
        output_dir,
        f"l2_analysis_equalIndifference_nsub_{n_sub}",
        "_contrast_id_02",
    )
    h8 = opj(
        result_dir,
        output_dir,
        f"l2_analysis_equalRange_nsub_{n_sub}",
        "_contrast_id_02",
    )
    h9 = opj(
        result_dir, output_dir, f"l2_analysis_groupComp_nsub_{n_sub}", "_contrast_id_02"
    )

    h = [h1, h2, h3, h4, h5, h6, h7, h8, h9]

    repro_unthresh = [
        opj(filename, "_change_filename_.nii") for i, filename in enumerate(h)
    ]  # Change filename with the filename of the final results

    repro_thresh = [
        opj(filename, "_change_filename_.nii") for i, filename in enumerate(h)
    ]

    if not os.path.isdir(opj(result_dir, "NARPS-reproduction")):
        os.mkdir(opj(result_dir, "NARPS-reproduction"))

    for i, filename in enumerate(repro_unthresh):
        f_in = filename
        f_out = opj(
            result_dir,
            "NARPS-reproduction",
            f"team_{team_ID}_nsub_{n_sub}_hypo{i+1}_unthresholded.nii",
        )
        shutil.copyfile(f_in, f_out)

    for i, filename in enumerate(repro_thresh):
        f_in = filename
        f_out = opj(
            result_dir,
            "NARPS-reproduction",
            f"team_{team_ID}_nsub_{n_sub}_hypo{i+1}_thresholded.nii",
        )
        shutil.copyfile(f_in, f_out)

    print(f"Results files of team {team_ID} reorganized.")
