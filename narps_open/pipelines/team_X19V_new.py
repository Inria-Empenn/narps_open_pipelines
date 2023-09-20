#!/usr/bin/python
# coding: utf-8
"""Classes and functions for the pipeline of team X19V."""


import os
import shutil
from os.path import join as opj
from pathlib import Path

import numpy as np
import pandas as pd
from nipype import Node, Workflow
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces.base import Bunch
from nipype.interfaces.fsl import BET, FILMGLS, FEATModel, IsotropicSmooth, Level1Design
from nipype.interfaces.io import DataSink, SelectFiles
from nipype.interfaces.utility import Function, IdentityInterface

from narps_open.pipelines import Pipeline


class PipelineTeamX19V(Pipeline):
    """A class that defines the pipeline of team X19V."""

    def __init__(self, bids_dir: str | Path, subject_list: str | list[str]):
        """Create a pipeline object."""
        super().__init__()
        self.fwhm = 5.0
        self.team_id = "X19V"
        self.contrast_list = ["0001", "0002", "0003"]

        self.directories.dataset_dir = str(bids_dir)

        if isinstance(subject_list, str):
            subject_list = [subject_list]
        self.subject_list = subject_list

    def get_hypotheses_outputs():
        """Get output for each hypothesis.

        Not yet implemented.
        """
        ...

    def get_preprocessing():
        """Create preprocessing workflow.

        Unused by this team.
        """
        ...

    # [INFO] There was no run level analysis for the pipelines using FSL
    def get_run_level_analysis(self):
        """Return a Nipype workflow describing the run level analysis part of the pipeline."""
        return None

    # [INFO] This function is used in the subject level analysis pipelines using FSL
    def get_subject_infos(self, event_file: str) -> list[type[Bunch]]:
        """Create Bunchs for specifyModel.

        Parameters :
        - event_file : str, file corresponding to the run and the subject to analyze

        Returns :
        - subject_info : list of Bunch for 1st level analysis.
        """
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
                    onset[c].append(float(info[0]))
                    duration[c].append(float(4))
                    if c == "gain":
                        amplitude[c].append(float(info[2]))
                    elif c == "loss":
                        amplitude[c].append(float(info[3]))
                    elif c == "trial":
                        amplitude[c].append(float(1))

        amplitude["gain"] = amplitude["gain"] - np.mean(amplitude["gain"])
        amplitude["loss"] = amplitude["loss"] - np.mean(amplitude["loss"])

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

    def get_parameters_file(
        self,
        file: str | Path,
        subject_id: str,
        run_id: str,
    ):
        """Create new tsv files with only desired parameters per subject per run.

        Parameters :
        - filepaths : paths to subject parameters file (i.e. one per run)
        - subject_id : subject for whom the 1st level analysis is made
        - run_id: run for which the 1st level analysis is made
        - result_dir: str, directory where results will be stored
        - working_dir: str, name of the sub-directory for intermediate results

        Return :
        - parameters_file : paths to new files containing only desired parameters.
        """
        df = pd.read_csv(file, sep="\t", header=0)
        temp_list = np.array(
            [df["X"], df["Y"], df["Z"], df["RotX"], df["RotY"], df["RotZ"]]
        )  # Parameters we want to use for the model
        retained_parameters = pd.DataFrame(np.transpose(temp_list))

        paramater_files_dir = Path(
            self.directories.results_dir
            / self.directories.working_dir
            / "parameters_file"
        )
        paramater_files_dir.mkdir(parents=True, exist_ok=True)

        new_path = (
            paramater_files_dir / f"parameters_file_sub-{subject_id}_run{run_id}.tsv"
        )
        retained_parameters.to_csv(
            new_path, sep="\t", index=False, header=False, na_rep="0.0"
        )

        parameters_file = [new_path]
        return parameters_file

    # [INFO] Linear contrast effects: 'Gain' vs. baseline, 'Loss' vs. baseline.
    def get_contrasts(self) -> list[tuple]:
        """Create the list of tuples that represents contrasts.

        Each contrast is in the form :
        (Name, Stat, [list of condition names], [weights on those conditions])

        Parameters:
            - subject_id: str, ID of the subject

        Returns:
            - contrasts: list of tuples, list of contrasts to analyze
        """
        # list of condition names
        conditions = ["trial", "gain", "loss"]

        # create contrasts
        gain = ("gain", "T", conditions, [0, 1, 0])

        loss = ("loss", "T", conditions, [0, 0, 1])

        gain_sup = ("gain_sup_loss", "T", conditions, [0, 1, -1])

        loss_sup = ("loss_sup_gain", "T", conditions, [0, -1, 1])

        # contrast list
        contrasts = [gain, loss, gain_sup, loss_sup]

        return contrasts

    def get_subject_level_analysis(
        self,
        exp_dir: str | Path,
        result_dir: str | Path,
        working_dir: str | Path,
        output_dir: str,
    ):
        """Return the first level analysis workflow.

        Parameters:
                - exp_dir: str, directory where raw data are stored
                - result_dir: str, directory where results will be stored
                - working_dir: str, name of the sub-directory for intermediate results
                - output_dir: str, name of the sub-directory for final results

        Returns:
                - l1_analysis : Nipype WorkFlow
        """
        exp_dir = str(exp_dir)
        result_dir = str(result_dir)
        working_dir = str(working_dir)

        # Infosource Node - To iterate on subject and runs
        infosource = Node(
            IdentityInterface(fields=["subject_id", "run_id"]), name="infosource"
        )
        infosource.iterables = [
            ("subject_id", self.subject_list),
            ("run_id", self.run_list),
        ]

        # Templates to select files node
        func_file = opj(
            "derivatives",
            "fmriprep",
            "sub-{subject_id}",
            "func",
            "sub-{subject_id}_task-MGT_run-{run_id}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz",
        )

        event_file = opj(
            "sub-{subject_id}",
            "func",
            "sub-{subject_id}_task-MGT_run-{run_id}_events.tsv",
        )

        param_file = opj(
            "derivatives",
            "fmriprep",
            "sub-{subject_id}",
            "func",
            "sub-{subject_id}_task-MGT_run-{run_id}_bold_confounds.tsv",
        )

        template = {"func": func_file, "event": event_file, "param": param_file}

        # SelectFiles node - to select necessary files
        selectfiles = Node(
            SelectFiles(template, base_directory=exp_dir), name="selectfiles"
        )

        # DataSink Node - store the wanted results in the wanted repository
        datasink = Node(
            DataSink(base_directory=result_dir, container=output_dir), name="datasink"
        )

        # Skullstripping
        skullstrip = Node(BET(frac=0.1, functional=True, mask=True), name="skullstrip")

        # Smoothing
        smooth = Node(IsotropicSmooth(fwhm=self.fwhm), name="smooth")

        # Node contrasts to get contrasts
        contrasts = Node(
            Function(
                function=self.get_contrasts,
                input_names=["subject_id"],
                output_names=["contrasts"],
            ),
            name="contrasts",
        )

        # Get Subject Info - get subject specific condition information
        get_subject_infos = Node(
            Function(
                input_names=["event_file"],
                output_names=["subject_info"],
                function=self.get_subject_infos,
            ),
            name="get_subject_infos",
        )

        specify_model = Node(
            SpecifyModel(
                high_pass_filter_cutoff=60, input_units="secs", time_repetition=self.tr
            ),
            name="specify_model",
        )

        parameters = Node(
            Function(
                function=self.get_parameters_file,
                input_names=[
                    "file",
                    "subject_id",
                    "run_id",
                    "result_dir",
                    "working_dir",
                ],
                output_names=["parameters_file"],
            ),
            name="parameters",
        )

        parameters.inputs.result_dir = result_dir
        parameters.inputs.working_dir = working_dir

        # First temporal derivatives of the two regressors were also used,
        # along with temporal filtering (60 s) of all the independent variable time-series.
        # No motion parameter regressors used.

        l1_design = Node(
            Level1Design(
                bases={"dgamma": {"derivs": False}},
                interscan_interval=self.tr,
                model_serial_correlations=True,
            ),
            name="l1_design",
        )

        model_generation = Node(FEATModel(), name="model_generation")

        model_estimate = Node(FILMGLS(), name="model_estimate")

        remove_smooth = Node(
            Function(
                input_names=[
                    "subject_id",
                    "run_id",
                    "files",
                    "result_dir",
                    "working_dir",
                ],
                function=rm_smoothed_files,
            ),
            name="remove_smooth",
        )

        remove_smooth.inputs.result_dir = result_dir
        remove_smooth.inputs.working_dir = working_dir

        # Create l1 analysis workflow and connect its nodes
        l1_analysis = Workflow(
            base_dir=opj(result_dir, working_dir), name="l1_analysis"
        )

        l1_analysis.connect(
            [
                (
                    infosource,
                    selectfiles,
                    [("subject_id", "subject_id"), ("run_id", "run_id")],
                ),
                (selectfiles, get_subject_infos, [("event", "event_file")]),
                (selectfiles, parameters, [("param", "file")]),
                (infosource, contrasts, [("subject_id", "subject_id")]),
                (
                    infosource,
                    parameters,
                    [("subject_id", "subject_id"), ("run_id", "run_id")],
                ),
                (selectfiles, skullstrip, [("func", "in_file")]),
                (skullstrip, smooth, [("out_file", "in_file")]),
                (
                    parameters,
                    specify_model,
                    [("parameters_file", "realignment_parameters")],
                ),
                (smooth, specify_model, [("out_file", "functional_runs")]),
                (get_subject_infos, specify_model, [("subject_info", "subject_info")]),
                (contrasts, l1_design, [("contrasts", "contrasts")]),
                (specify_model, l1_design, [("session_info", "session_info")]),
                (
                    l1_design,
                    model_generation,
                    [("ev_files", "ev_files"), ("fsf_files", "fsf_file")],
                ),
                (smooth, model_estimate, [("out_file", "in_file")]),
                (
                    model_generation,
                    model_estimate,
                    [("design_file", "design_file")],
                ),
                (
                    infosource,
                    remove_smooth,
                    [("subject_id", "subject_id"), ("run_id", "run_id")],
                ),
                (model_estimate, remove_smooth, [("results_dir", "files")]),
                (model_estimate, datasink, [("results_dir", "l1_analysis.@results")]),
                (
                    model_generation,
                    datasink,
                    [
                        ("design_file", "l1_analysis.@design_file"),
                        ("design_image", "l1_analysis.@design_img"),
                    ],
                ),
                (skullstrip, datasink, [("mask_file", "l1_analysis.@skullstriped")]),
            ]
        )

        return l1_analysis

    # [INFO] This function returns the list of ids and files of each group of participants
    # to do analyses for both groups, and one between the two groups.
    def get_subgroups_contrasts(
        copes, varcopes, subject_list: list, participants_file: str
    ):
        """Return the file list containing only the files belonging to subject in the wanted group.

        Parameters :
        - copes: original file list selected by select_files node
        - varcopes: original file list selected by select_files node
        - subject_ids: list of subject IDs that are analyzed
        - participants_file: file containing participants characteristics

        Returns :
        - copes_equal_indifference : a subset of copes corresponding to subjects
        in the equalIndifference group
        - copes_equal_range : a subset of copes corresponding to subjects
        in the equalRange group
        - copes_global : a list of all copes
        - varcopes_equal_indifference : a subset of varcopes corresponding to subjects
        in the equalIndifference group
        - varcopes_equal_range : a subset of varcopes corresponding to subjects
        in the equalRange group
        - equal_indifference_id : a list of subject ids in the equalIndifference group
        - equal_range_id : a list of subject ids in the equalRange group
        - varcopes_global : a list of all varcopes
        """
        equal_range_id = []
        equal_indifference_id = []

        # Reading file containing participants IDs and groups
        with open(participants_file, "rt") as file:
            next(file)  # skip the header

            for line in file:
                info = line.strip().split()

                # Checking for each participant if its ID was selected
                # and separate people depending on their group
                if info[0][-3:] in subject_list and info[1] == "equalIndifference":
                    equal_indifference_id.append(info[0][-3:])
                elif info[0][-3:] in subject_list and info[1] == "equalRange":
                    equal_range_id.append(info[0][-3:])

        copes_equal_indifference = []
        copes_equal_range = []
        copes_global = []
        varcopes_equal_indifference = []
        varcopes_equal_range = []
        varcopes_global = []

        # Checking for each selected file if the corresponding participant was selected
        # and add the file to the list corresponding to its group
        for cope, varcope in zip(copes, varcopes):
            sub_id = cope.split("/")
            if sub_id[-2][-3:] in equal_indifference_id:
                copes_equal_indifference.append(cope)
            elif sub_id[-2][-3:] in equal_range_id:
                copes_equal_range.append(cope)
            if sub_id[-2][-3:] in subject_list:
                copes_global.append(cope)

            sub_id = varcope.split("/")
            if sub_id[-2][-3:] in equal_indifference_id:
                varcopes_equal_indifference.append(varcope)
            elif sub_id[-2][-3:] in equal_range_id:
                varcopes_equal_range.append(varcope)
            if sub_id[-2][-3:] in subject_list:
                varcopes_global.append(varcope)

        return (
            copes_equal_indifference,
            copes_equal_range,
            varcopes_equal_indifference,
            varcopes_equal_range,
            equal_indifference_id,
            equal_range_id,
            copes_global,
            varcopes_global,
        )

    # [INFO] This function creates the dictionary of regressors used in FSL Nipype pipelines
    def get_regressors(
        equal_range_id: list,
        equal_indifference_id: list,
        method: str,
        subject_list: list,
    ) -> dict:
        """Create dictionary of regressors for group analysis.

        Parameters:
            - equal_range_id: ids of subjects in equal range group
            - equal_indifference_id: ids of subjects in equal indifference group
            - method: one of "equalRange", "equalIndifference" or "groupComp"
            - subject_list: ids of subject for which to do the analysis

        Returns:
            - regressors: regressors used to distinguish groups in FSL group analysis
        """
        # For one sample t-test, creates a dictionary
        # with a list of the size of the number of participants
        if method == "equalRange":
            regressors = dict(group_mean=[1 for i in range(len(equal_range_id))])
        elif method == "equalIndifference":
            regressors = dict(group_mean=[1 for i in range(len(equal_indifference_id))])

        # For two sample t-test, creates 2 lists:
        #  - one for equal range group,
        #  - one for equal indifference group
        # Each list contains n_sub values with 0 and 1 depending on the group of the participant
        # For equalRange_reg list --> participants with a 1 are in the equal range group
        elif method == "groupComp":
            equalRange_reg = [
                1 for i in range(len(equal_range_id) + len(equal_indifference_id))
            ]
            equalIndifference_reg = [
                0 for i in range(len(equal_range_id) + len(equal_indifference_id))
            ]

            for index, subject_id in enumerate(subject_list):
                if subject_id in equal_indifference_id:
                    equalIndifference_reg[index] = 1
                    equalRange_reg[index] = 0

            regressors = dict(
                equalRange=equalRange_reg, equalIndifference=equalIndifference_reg
            )

        return regressors

    def get_group_level_analysis(self):
        """Return all workflows for the group level analysis.

        Returns;
            - a list of nipype.WorkFlow
        """
        methods = ["equalRange", "equalIndifference", "groupComp"]
        return [
            self.get_group_level_analysis_sub_workflow(method) for method in methods
        ]

    def get_group_level_analysis_sub_workflow(self, method):
        """Return a workflow for the group level analysis.

        Parameters:
            - method: one of 'equalRange', 'equalIndifference' or 'groupComp'

        Returns:
            - group_level_analysis: nipype.WorkFlow
        """
        # [INFO] The following part stays the same for all preprocessing pipelines

        # Infosource node - iterate over the list of contrasts generated
        # by the subject level analysis
        info_source = Node(
            IdentityInterface(
                fields=["contrast_id", "subjects"], subjects=self.subject_list
            ),
            name="info_source",
        )
        info_source.iterables = [("contrast_id", self.contrast_list)]

        # Templates to select files node
        # [TODO] Change the name of the files depending on the filenames
        # of results of first level analysis
        template = {
            "cope": opj(
                self.directories.results_dir,
                "subject_level_analysis",
                "_contrast_id_{contrast_id}_subject_id_*",
                "cope1.nii.gz",
            ),
            "varcope": opj(
                self.directories.results_dir,
                "subject_level_analysis",
                "_contrast_id_{contrast_id}_subject_id_*",
                "varcope1.nii.gz",
            ),
            "participants": opj(self.directories.dataset_dir, "participants.tsv"),
        }
        select_files = Node(
            SelectFiles(
                templates,
                base_directory=self.directories.results_dir,
                force_list=True,
            ),
            name="select_files",
        )

        # Datasink node - to save important files
        data_sink = Node(
            DataSink(base_directory=self.directories.output_dir),
            name="data_sink",
        )

        contrasts = Node(
            Function(
                input_names=[
                    "copes",
                    "varcopes",
                    "subject_ids",
                    "participants_file",
                ],
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
                function=self.get_subgroups_contrasts,
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
                function=self.get_regressors,
            ),
            name="regs",
        )
        regs.inputs.method = method
        regs.inputs.subject_list = subject_list

        # [INFO] The following part has to be modified with nodes of the pipeline

        # [TODO] For each node, replace 'node_name' by an explicit name, and use it for both:
        #   - the name of the variable in which you store the Node object
        #   - the 'name' attribute of the Node
        # [TODO] The node_function refers to a NiPype interface that you must import
        # at the beginning of the file.
        node_name = Node(node_function, name="node_name")

        # [INFO] The following part defines the nipype workflow and the connections between nodes

        # Compute the number of participants used to do the analysis
        nb_subjects = len(self.subject_list)

        # Declare the workflow
        group_level_analysis = Workflow(
            base_dir=self.directories.working_dir,
            name=f"group_level_analysis_{method}_nsub_{nb_subjects}",
        )
        group_level_analysis.connect(
            [
                (
                    info_source,
                    select_files,
                    [("contrast_id", "contrast_id")],
                ),
                (
                    info_source,
                    subgroups_contrasts,
                    [("subject_list", "subject_ids")],
                ),
                (
                    select_files,
                    subgroups_contrasts,
                    [
                        ("cope", "copes"),
                        ("varcope", "varcopes"),
                        ("participants", "participants_file"),
                    ],
                ),
                (
                    select_files,
                    node_name[("func", "node_input_name")],
                ),
                (
                    node_variable,
                    datasink_groupanalysis,
                    [("node_output_name", "preprocess.@sym_link")],
                ),
            ]
        )  # Complete with other links between nodes

        # [INFO] Here we define the contrasts used for the group level analysis, depending on the
        # method used.
        if method in ("equalRange", "equalIndifference"):
            contrasts = [
                ("Group", "T", ["mean"], [1]),
                ("Group", "T", ["mean"], [-1]),
            ]

        elif method == "groupComp":
            contrasts = [
                (
                    "Eq range vs Eq indiff in loss",
                    "T",
                    ["Group_{1}", "Group_{2}"],
                    [1, -1],
                )
            ]

        # [INFO] Here we simply return the created workflow
        return group_level_analysis


def rm_smoothed_files(
    subject_id: str, run_id: str, result_dir: str | Path, working_dir: str
):
    if isinstance(result_dir, str):
        result_dir = Path(result_dir)
    smooth_dir = (
        result_dir
        / working_dir
        / "l1_analysis"
        / f"_run_id_{run_id}_subject_id_{subject_id}"
        / "smooth"
    )

    try:
        shutil.rmtree(smooth_dir)
    except OSError as e:
        print(e)
    else:
        print("The directory was deleted successfully")
