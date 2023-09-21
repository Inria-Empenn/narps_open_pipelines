#!/usr/bin/python
# coding: utf-8

from __future__ import annotations

from os.path import join
from pathlib import Path

import pandas as pd

from nipype import Node, Workflow  # , JoinNode, MapNode
from nipype.algorithms.misc import Gunzip
from nipype.interfaces.base import Bunch
from nipype.interfaces.io import DataSink, SelectFiles

# [INFO] a list of SPM-specific interfaces
# from nipype.algorithms.modelgen import SpecifySPMModel
# from nipype.interfaces.spm import (
#    Level1Design, OneSampleTTestDesign, TwoSampleTTestDesign,
#    EstimateModel, EstimateContrast, Threshold
#    )
from nipype.interfaces.spm import Smooth
from nipype.interfaces.utility import Function, IdentityInterface

from narps_open.data.description import TeamDescription
from narps_open.pipelines import Pipeline

TEAM_ID = "R9K3"
DESC = TeamDescription(TEAM_ID)


class PipelineTeamR9K3(Pipeline):
    """A class that defines the pipeline of team R9K3"""

    def __init__(self,  bids_dir: str | Path, subject_list: str | list[str] | None = None):
        super().__init__()

        self.fwhm = float(DESC.derived["func_fwhm"])
        
        self.team_id = TEAM_ID

        self.directories.dataset_dir = str(bids_dir)

        if subject_list is None:
            participant_tsv = Path(self.directories.dataset_dir) / "participants.tsv"
            df = pd.read_csv(participant_tsv, sep="\t")
            subject_list = df["participant_id"].tolist()
        if isinstance(subject_list, str):
            subject_list = [subject_list]
        excluded_participants = DESC.derived["excluded_participants"]
        if excluded_participants != "n/a":
            excluded_participants = excluded_participants.split(",")
            excluded_participants = [s.strip() for s in excluded_participants]
            excluded_participants = [s.strip()[-3:] for s in excluded_participants]
            print(f"Excluding participants: {excluded_participants}")
        self.subject_list = [s for s in subject_list if s not in excluded_participants]

    def get_preprocessing(self):
        """Smooth the fmriprep data."""
        # IdentityInterface node - allows to iterate over subjects and runs
        info_source = Node(
            IdentityInterface(fields=["subject_id", "run_id"]), name="info_source"
        )
        info_source.iterables = [
            ("subject_id", self.subject_list),
            ("run_id", self.run_list),
        ]

        # Templates to select files node
        file_templates = {
            "anat": join("sub-{subject_id}", "anat", "sub-{subject_id}_T1w.nii.gz"),
            "func": join(
                "sub-{subject_id}",
                "func",
                "sub-{subject_id}_task-MGT_run-{run_id}_bold.nii.gz",
            ),
            "magnitude": join(
                "sub-{subject_id}", "fmap", "sub-{subject_id}_magnitude1.nii.gz"
            ),
            "phasediff": join(
                "sub-{subject_id}", "fmap", "sub-{subject_id}_phasediff.nii.gz"
            ),
        }

        # Nodes
        select_files = Node(
            SelectFiles(file_templates, base_directory=self.directories.dataset_dir),
            name="select_files",
        )

        data_sink = Node(
            DataSink(base_directory=self.directories.output_dir),
            name="data_sink",
        )

        smooth = Node(Smooth(fwhm=self.fwhm, implicit_masking=False), name="smooth")

        gunzip = Node(Gunzip(), name="gunzip")

        # Connect workflow nodes
        preprocessing = Workflow(
            base_dir=self.directories.working_dir, name="preprocessing"
        )

        preprocessing.connect(
            [
                (
                    info_source,
                    select_files,
                    [("subject_id", "subject_id"), ("run_id", "run_id")],
                ),
                (
                    select_files,
                    gunzip,
                    [("func", "in_file")],
                ),
                (
                    gunzip,
                    smooth,
                    [("out_file", "in_files")],
                ),
                (
                    smooth,
                    data_sink,
                    [("smoothed_files", "preprocessing.@sym_link")],
                ),
            ]
        )

        return preprocessing

    # [INFO] There was no run level analysis for the pipelines using SPM
    def get_run_level_analysis(self):
        """Return a Nipype worflow describing the run level analysis part of the pipeline"""
        return None

    """
    In first level analyses, three regressors of interest were included in the model: 

    1) task-related activity, 
    2) task-related activity modulated by the amount of gain, and 
    3) task-related activity modulated by the amount of loss. 

    The regressor for the basic task-related activity was defined 
    as a vector of ones at every trial onset and zeros at all other time points 
    (event duration = 0 s). 

    Within each functional run, each parametric modulator 
    (the raw gain or loss values) was scaled such that its maximum value became 1. 
    All three regressors were convolved with the canonical hemodynamic response function. 
    The two modulated regressors were orthogonalized against the original task regressor. 
    Temporal/dispersion derivatives of the regressors were not included. 

    In both first and second level analyses, model parameters were estimated 
    using the classical (Restricted Maximum Likelihood) method as implemented in SPM12. 

    For second level analyses, we used a mixed-effects approach with weighted least squares. 

    In the second level analysis performed to test hypothesis 9 
    (two-sample t-test comparing the equal indifference group and the equal range group), 
    the two groups were assumed to be independent and have unequal variance.   
    """

    def get_subject_infos(event_files: list, runs: list):
        """
        Create Bunchs for specifySPMModel.

        Parameters :
        - event_files: list of events files (one per run) for the subject
        - runs: list of runs to use

        Returns :
        - subject_info : list of Bunch for 1st level analysis.
        """

        condition_names = ["trial", "accepting", "rejecting"]
        onset = {}
        duration = {}
        weights_gain = {}
        weights_loss = {}

        # Loop over number of runs
        for run_id in range(len(runs)):
            # Create dictionary items with empty lists
            onset.update({s + "_run" + str(run_id + 1): [] for s in condition_names})
            duration.update({s + "_run" + str(run_id + 1): [] for s in condition_names})
            weights_gain.update({"gain_run" + str(run_id + 1): []})
            weights_loss.update({"loss_run" + str(run_id + 1): []})

            with open(event_files[run_id], "rt") as event_file:
                next(event_file)  # skip the header

                for line in event_file:
                    info = line.strip().split()

                    for condition in condition_names:
                        val = (
                            condition + "_run" + str(run_id + 1)
                        )  # trial_run1 or accepting_run1
                        val_gain = "gain_run" + str(run_id + 1)  # gain_run1
                        val_loss = "loss_run" + str(run_id + 1)  # loss_run1
                        if condition == "trial":
                            onset[val].append(float(info[0]))  # onsets for trial_run1
                            duration[val].append(float(4))
                            weights_gain[val_gain].append(float(info[2]))
                            weights_loss[val_loss].append(float(info[3]))
                        elif condition == "accepting" and "accept" in info[5]:
                            onset[val].append(float(info[0]) + float(info[4]))
                            duration[val].append(float(0))
                        elif condition == "rejecting" and "reject" in info[5]:
                            onset[val].append(float(info[0]) + float(info[4]))
                            duration[val].append(float(0))

        # Bunching is done per run, i.e. trial_run1, trial_run2, etc.
        # But names must not have '_run1' etc because we concatenate runs
        subject_info = []
        for run_id in range(len(runs)):
            conditions = [s + "_run" + str(run_id + 1) for s in condition_names]
            gain = "gain_run" + str(run_id + 1)
            loss = "loss_run" + str(run_id + 1)

            subject_info.insert(
                run_id,
                Bunch(
                    conditions=condition_names,
                    onsets=[onset[c] for c in conditions],
                    durations=[duration[c] for c in conditions],
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

    # [INFO] This function creates the contrasts that will be analyzed in the first level analysis
    # [TODO] Adapt this example to your specific pipeline
    def get_contrasts():
        """
        Create the list of tuples that represents contrasts.
        Each contrast is in the form :
        (Name,Stat,[list of condition names],[weights on those conditions])

        Returns:
            - contrasts: list of tuples, list of contrasts to analyze
        """
        # List of condition names
        conditions = ["trial", "trialxgain^1", "trialxloss^1"]

        # Create contrasts
        trial = ("trial", "T", conditions, [1, 0, 0])
        effect_gain = ("effect_of_gain", "T", conditions, [0, 1, 0])
        effect_loss = ("effect_of_loss", "T", conditions, [0, 0, 1])

        # Contrast list
        return [trial, effect_gain, effect_loss]

    def get_subject_level_analysis(self):
        """Return a Nipype worflow describing the subject level analysis part of the pipeline"""

        # [INFO] The following part stays the same for all pipelines

        # Infosource Node - To iterate on subjects
        info_source = Node(
            IdentityInterface(
                fields=[
                    "subject_id",
                    "dataset_dir",
                    "results_dir",
                    "working_dir",
                    "run_list",
                ],
                dataset_dir=self.directories.dataset_dir,
                results_dir=self.directories.results_dir,
                working_dir=self.directories.working_dir,
                run_list=self.run_list,
            ),
            name="info_source",
        )
        info_source.iterables = [("subject_id", self.subject_list)]

        # Templates to select files node
        # [TODO] Change the name of the files depending on the filenames of results of preprocessing
        templates = {
            "func": join(
                self.directories.results_dir,
                "preprocess",
                "_run_id_*_subject_id_{subject_id}",
                "complete_filename_{subject_id}_complete_filename.nii",
            ),
            "event": join(
                self.directories.dataset_dir,
                "sub-{subject_id}",
                "func",
                "sub-{subject_id}_task-MGT_run-*_events.tsv",
            ),
        }

        # SelectFiles node - to select necessary files
        select_files = Node(
            SelectFiles(templates, base_directory=self.directories.dataset_dir),
            name="select_files",
        )

        # DataSink Node - store the wanted results in the wanted repository
        data_sink = Node(
            DataSink(base_directory=self.directories.output_dir), name="data_sink"
        )

        # [INFO] This is the node executing the get_subject_infos_spm function
        # Subject Infos node - get subject specific condition information
        subject_infos = Node(
            Function(
                input_names=["event_files", "runs"],
                output_names=["subject_info"],
                function=self.get_subject_infos,
            ),
            name="subject_infos",
        )
        subject_infos.inputs.runs = self.run_list

        # [INFO] This is the node executing the get_contrasts function
        # Contrasts node - to get contrasts
        contrasts = Node(
            Function(
                input_names=["subject_id"],
                output_names=["contrasts"],
                function=self.get_contrasts,
            ),
            name="contrasts",
        )

        # [INFO] The following part has to be modified with nodes of the pipeline

        # [TODO] For each node, replace 'node_name' by an explicit name, and use it for both:
        #   - the name of the variable in which you store the Node object
        #   - the 'name' attribute of the Node
        # [TODO] The node_function refers to a NiPype interface that you must import
        # at the begining of the file.
        node_name = Node(node_function, name="node_name")

        # [TODO] Add other nodes with the different steps of the pipeline

        # [INFO] The following part defines the nipype workflow and the connections between nodes

        subject_level_analysis = Workflow(
            base_dir=self.directories.working_dir, name="subject_level_analysis"
        )
        # [TODO] Add the connections the workflow needs
        # [INFO] Input and output names can be found on NiPype documentation
        subject_level_analysis.connect(
            [
                (info_source, select_files, [("subject_id", "subject_id")]),
                (info_source, contrasts, [("subject_id", "subject_id")]),
                (select_files, subject_infos, [("event", "event_files")]),
                (select_files, node_name, [("func", "node_input_name")]),
                (node_name, data_sink, [("node_output_name", "preprocess.@sym_link")]),
            ]
        )

        # [INFO] Here we simply return the created workflow
        return subject_level_analysis

    # [INFO] This function returns the list of ids and files of each group of participants
    # to do analyses for both groups, and one between the two groups.
    def get_subset_contrasts(file_list, subject_list: list, participants_file: str):
        """
        This function return the file list containing only the files belonging
        to the subjects in the wanted group.

        Parameters :
        - file_list : original file list selected by selectfiles node
        - subject_list : list of subject IDs that are in the wanted group for the analysis
        - participants_file: str, file containing participants caracteristics

        Returns :
        - equal_indifference_id : a list of subject ids in the equalIndifference group
        - equal_range_id : a list of subject ids in the equalRange group
        - equal_indifference_files : a subset of file_list corresponding to subjects
        in the equalIndifference group
        - equal_range_files : a subset of file_list corresponding to subjects
        in the equalRange group
        """
        equal_indifference_id = []
        equal_range_id = []
        equal_indifference_files = []
        equal_range_files = []

        # Reading file containing participants IDs and groups
        with open(participants_file, "rt") as file:
            next(file)  # skip the header
            for line in file:
                info = line.strip().split()
                if info[0][-3:] in subject_list and info[1] == "equalIndifference":
                    equal_indifference_id.append(info[0][-3:])
                elif info[0][-3:] in subject_list and info[1] == "equalRange":
                    equal_range_id.append(info[0][-3:])

        for file in file_list:
            sub_id = file.split("/")
            if sub_id[-2][-3:] in equal_indifference_id:
                equal_indifference_files.append(file)
            elif sub_id[-2][-3:] in equal_range_id:
                equal_range_files.append(file)

        return (
            equal_indifference_id,
            equal_range_id,
            equal_indifference_files,
            equal_range_files,
        )

    def get_group_level_analysis(self):
        """
        Return all workflows for the group level analysis.

        Returns;
            - a list of nipype.WorkFlow
        """

        methods = ["equalRange", "equalIndifference", "groupComp"]
        return [
            self.get_group_level_analysis_sub_workflow(method) for method in methods
        ]

    def get_group_level_analysis_sub_workflow(self, method):
        """
        Return a workflow for the group level analysis.

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
        templates = {
            "contrast": join(
                self.directories.results_dir,
                "subject_level_analysis",
                "_subject_id_*",
                "complete_filename_{contrast_id}_complete_filename.nii",
            ),
            "participants": join(self.directories.dataset_dir, "participants.tsv"),
        }
        select_files = Node(
            SelectFiles(
                templates, base_directory=self.directories.results_dir, force_list=True
            ),
            name="select_files",
        )

        # Datasink node - to save important files
        data_sink = Node(
            DataSink(base_directory=self.directories.output_dir),
            name="data_sink",
        )

        # Contrasts node - to select subset of contrasts
        sub_contrasts = Node(
            Function(
                input_names=[
                    "file_list",
                    "method",
                    "subject_list",
                    "participants_file",
                ],
                output_names=[
                    "equalIndifference_id",
                    "equalRange_id",
                    "equalIndifference_files",
                    "equalRange_files",
                ],
                function=self.get_subset_contrasts,
            ),
            name="sub_contrasts",
        )
        sub_contrasts.inputs.method = method

        # [INFO] The following part has to be modified with nodes of the pipeline

        # [TODO] For each node, replace 'node_name' by an explicit name, and use it for both:
        #   - the name of the variable in which you store the Node object
        #   - the 'name' attribute of the Node
        # [TODO] The node_function refers to a NiPype interface that you must import
        # at the begining of the file.
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
                (info_source, sub_contrasts, [("subjects", "subject_list")]),
                (
                    select_files,
                    sub_contrasts,
                    [("contrast", "file_list"), ("participants", "participants_file")],
                ),  # Complete with other links between nodes
            ]
        )

        # [INFO] Here whe define the contrasts used for the group level analysis, depending on the
        # method used.
        if method in ("equalRange", "equalIndifference"):
            contrasts = [("Group", "T", ["mean"], [1]), ("Group", "T", ["mean"], [-1])]

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

    def get_hypotheses_outputs(self):
        ...
