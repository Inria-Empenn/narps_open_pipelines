#!/usr/bin/python
# coding: utf-8
"""Classes and functions for the pipeline of team X19V."""

import os
import shutil
from os.path import join
from pathlib import Path

import nibabel as nb
import numpy as np
from nipype import MapNode, Node, Workflow
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces.base import Bunch
from nipype.interfaces.fsl import (
    BET,
    FILMGLS,
    FLAMEO,
    Cluster,
    FEATModel,
    IsotropicSmooth,
    L2Model,
    Level1Design,
    Merge,
    MultipleRegressDesign,
    SmoothEstimate,
)
from nipype.interfaces.io import DataSink, SelectFiles
from nipype.interfaces.utility import Function, IdentityInterface

from narps_open.pipelines import Pipeline

# Event-Related design, 4 second events, with parametric modulation based on amount gained or lost on each trial.
# One 'gain' regressor, and one 'loss' regressor.
# No RT modelling


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

    def get_preprocessing():
        """Create preprocessing workflow.

        Unused by this team.
        """
        ...

    def get_parameters_file(
        self,
        file: str | Path,
        subject_id: str,
        run_id: str,
    ) -> list[Path]:
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
        import numpy as np
        import pandas as pd

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

    def get_run_level_analysis(self):
        """Return the first level analysis workflow.

        Parameters:
                - result_dir: str, directory where results will be stored
                - working_dir: str, name of the sub-directory for intermediate results

        Returns:
                - l1_analysis : Nipype WorkFlow
        """
        # Infosource Node - To iterate on subject and runs
        infosource = Node(
            IdentityInterface(fields=["subject_id", "run_id"]), name="infosource"
        )
        infosource.iterables = [
            ("subject_id", self.subject_list),
            ("run_id", self.run_list),
        ]

        # Templates to select files node
        func_file = join(
            "derivatives",
            "fmriprep",
            "sub-{subject_id}",
            "func",
            "sub-{subject_id}_task-MGT_run-{run_id}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz",
        )
        event_file = join(
            "sub-{subject_id}",
            "func",
            "sub-{subject_id}_task-MGT_run-{run_id}_events.tsv",
        )
        param_file = join(
            "derivatives",
            "fmriprep",
            "sub-{subject_id}",
            "func",
            "sub-{subject_id}_task-MGT_run-{run_id}_bold_confounds.tsv",
        )
        template = {"func": func_file, "event": event_file, "param": param_file}

        # SelectFiles node - to select necessary files
        selectfiles = Node(
            SelectFiles(template, base_directory=str(self.directories.dataset_dir)),
            name="selectfiles",
        )

        # DataSink Node - store the wanted results in the wanted repository
        datasink = Node(
            DataSink(
                base_directory=str(self.directories.results_dir),
                container=self.directories.output_dir,
            ),
            name="datasink",
        )

        # Skullstripping
        skullstrip = Node(BET(frac=0.1, functional=True, mask=True), name="skullstrip")

        # Smoothing
        smooth = Node(IsotropicSmooth(fwhm=self.fwhm), name="smooth")

        # Node contrasts to get contrasts
        contrasts = Node(
            Function(
                function=get_contrasts,
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
                input_names=["file", "subject_id", "run_id"],
                output_names=["parameters_file"],
            ),
            name="parameters",
        )

        parameters.inputs.result_dir = self.directories.results_dir
        parameters.inputs.working_dir = self.directories.working_dir

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
                function=self.rm_smoothed_files,
            ),
            name="remove_smooth",
        )

        remove_smooth.inputs.result_dir = self.directories.results_dir
        remove_smooth.inputs.working_dir = self.directories.working_dir

        # Create l1 analysis workflow and connect its nodes
        l1_analysis = Workflow(
            base_dir=join(self.directories.results_dir, self.directories.working_dir),
            name="l1_analysis",
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

    def get_subject_level_analysis(self):
        """Return the 2nd level of analysis workflow.

        Returns:
        - l2_analysis: Nipype WorkFlow
        """
        # Infosource Node - To iterate on subject and runs
        infosource_2ndlevel = Node(
            IdentityInterface(fields=["subject_id", "contrast_id"]),
            name="infosource_2ndlevel",
        )
        infosource_2ndlevel.iterables = [
            ("subject_id", self.subject_list),
            ("contrast_id", self.contrast_list),
        ]

        # Templates to select files node
        copes_file = join(
            self.directories.output_dir,
            "l1_analysis",
            "_run_id_*_subject_id_{subject_id}",
            "results",
            "cope{contrast_id}.nii.gz",
        )
        varcopes_file = join(
            self.directories.output_dir,
            "l1_analysis",
            "_run_id_*_subject_id_{subject_id}",
            "results",
            "varcope{contrast_id}.nii.gz",
        )
        mask_file = join(
            self.directories.output_dir,
            "l1_analysis",
            "_run_id_*_subject_id_{subject_id}",
            "sub-{subject_id}_task-MGT_run-01_bold_space-MNI152NLin2009cAsym_preproc_brain_mask.nii.gz",
        )
        template = {"cope": copes_file, "varcope": varcopes_file, "mask": mask_file}

        # SelectFiles node - to select necessary files
        selectfiles_2ndlevel = Node(
            SelectFiles(template, base_directory=str(self.directories.results_dir)),
            name="selectfiles_2ndlevel",
        )

        datasink_2ndlevel = Node(
            DataSink(
                base_directory=str(self.directories.results_dir),
                container=self.directories.output_dir,
            ),
            name="datasink_2ndlevel",
        )

        # Generate design matrix
        specify_model_2ndlevel = Node(
            L2Model(num_copes=len(self.run_list)), name="l2model_2ndlevel"
        )

        # Merge copes and varcopes files for each subject
        merge_copes_2ndlevel = Node(Merge(dimension="t"), name="merge_copes_2ndlevel")
        merge_varcopes_2ndlevel = Node(
            Merge(dimension="t"), name="merge_varcopes_2ndlevel"
        )

        # Second level (single-subject, mean of all four scans) analyses: Fixed effects analysis.
        flame = Node(FLAMEO(run_mode="flame1"), name="flameo")

        l2_analysis = Workflow(
            base_dir=join(self.directories.results_dir, self.directories.working_dir),
            name="l2_analysis",
        )

        l2_analysis.connect(
            [
                (
                    infosource_2ndlevel,
                    selectfiles_2ndlevel,
                    [("subject_id", "subject_id"), ("contrast_id", "contrast_id")],
                ),
                (selectfiles_2ndlevel, merge_copes_2ndlevel, [("cope", "in_files")]),
                (
                    selectfiles_2ndlevel,
                    merge_varcopes_2ndlevel,
                    [("varcope", "in_files")],
                ),
                (selectfiles_2ndlevel, flame, [("mask", "mask_file")]),
                (merge_copes_2ndlevel, flame, [("merged_file", "cope_file")]),
                (merge_varcopes_2ndlevel, flame, [("merged_file", "var_cope_file")]),
                (
                    specify_model_2ndlevel,
                    flame,
                    [
                        ("design_mat", "design_file"),
                        ("design_con", "t_con_file"),
                        ("design_grp", "cov_split_file"),
                    ],
                ),
                (
                    flame,
                    datasink_2ndlevel,
                    [
                        ("zstats", "l2_analysis.@stats"),
                        ("tstats", "l2_analysis.@tstats"),
                        ("copes", "l2_analysis.@copes"),
                        ("var_copes", "l2_analysis.@varcopes"),
                    ],
                ),
            ]
        )

        return l2_analysis

    # [INFO] This function returns the list of ids and files of each group of participants
    # to do analyses for both groups, and one between the two groups.
    def get_subgroups_contrasts(
        self,
        copes: list[str],
        varcopes: list[str],
    ):
        """Return the list of ids and files of each group of participants \
           to do analyses for both groups, and one between the two groups.

        Parameters :
        - copes: original file list selected by selectfiles node
        - varcopes: original file list selected by selectfiles node

        This function return the file list containing
        only the files belonging to subject in the wanted group.
        """
        equalRange_id = []
        equalIndifference_id = []

        subject_list = [f"sub-{str(i)}" for i in self.subject_list]

        participants_file = Path(self.directories.dataset_dir) / "participants.tsv"

        with open(participants_file, "rt") as f:
            next(f)  # skip the header

            for line in f:
                info = line.strip().split()

                if info[0] in subject_list:
                    subject_label = info[0][-3:]

                    if info[1] == "equalIndifference":
                        equalIndifference_id.append(subject_label)
                    elif info[1] == "equalRange":
                        equalRange_id.append(subject_label)

        copes_equalIndifference = []
        copes_equalRange = []
        varcopes_equalIndifference = []
        varcopes_equalRange = []

        for file in copes:
            sub_id = file.split("/")
            subject_label = sub_id[-2][-3:]
            if subject_label in equalIndifference_id:
                copes_equalIndifference.append(file)
            elif subject_label in equalRange_id:
                copes_equalRange.append(file)

        for file in varcopes:
            sub_id = file.split("/")
            subject_label = sub_id[-2][-3:]
            if subject_label in equalIndifference_id:
                varcopes_equalIndifference.append(file)
            elif subject_label in equalRange_id:
                varcopes_equalRange.append(file)

        print(len(equalRange_id))
        print(len(equalIndifference_id))
        print(len(copes_equalIndifference))
        print(len(copes_equalRange))

        copes_global = copes_equalIndifference + copes_equalRange
        varcopes_global = varcopes_equalIndifference + varcopes_equalRange

        return (
            copes_equalIndifference,
            copes_equalRange,
            copes_global,
            varcopes_equalIndifference,
            varcopes_equalRange,
            varcopes_global,
            equalIndifference_id,
            equalRange_id,
        )

    # [INFO] This function creates the dictionary of regressors used in FSL Nipype pipelines
    def get_regressors(
        self,
        equal_range_id: list,
        equal_indifference_id: list,
        method: str,
    ) -> dict:
        """Create dictionary of regressors for group analysis.

        Parameters:
            - equal_range_id: ids of subjects in equal range group
            - equal_indifference_id: ids of subjects in equal indifference group
            - method: one of "equalRange", "equalIndifference" or "groupComp"

        Returns:
            - regressors: regressors used to distinguish groups in FSL group analysis
        """
        # For one sample t-test, creates a dictionary
        # with a list of the size of the number of participants
        if method == "equalRange":
            regressors = dict(group_mean=[1 for _ in range(len(equal_range_id))])
        elif method == "equalIndifference":
            regressors = dict(group_mean=[1 for _ in range(len(equal_indifference_id))])

        # For two sample t-test, creates 2 lists:
        #  - one for equal range group,
        #  - one for equal indifference group
        # Each list contains n_sub values with 0 and 1 depending on the group of the participant
        # For equalRange_reg list --> participants with a 1 are in the equal range group
        elif method == "groupComp":
            equalRange_reg = [
                1 for _ in range(len(equal_range_id) + len(equal_indifference_id))
            ]
            equalIndifference_reg = [
                0 for _ in range(len(equal_range_id) + len(equal_indifference_id))
            ]

            for index, subject_id in enumerate(self.subject_list):
                if subject_id in equal_indifference_id:
                    equalIndifference_reg[index] = 1
                    equalRange_reg[index] = 0

            regressors = dict(
                equalRange=equalRange_reg, equalIndifference=equalIndifference_reg
            )

        return regressors

    def rm_smoothed_files(self, subject_id: str, run_id: str):
        """Remove directories containting smoothed files."""
        smooth_dir = (
            Path(self.directories.results_dir)
            / self.directories.working_dir
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

    def get_group_level_analysis(
        self,
        method,
    ):
        """Return the group level of analysis workflow.

        Parameters:
                - method: one of "equalRange", "equalIndifference" or "groupComp"


        Returns:
                - l3_analysis: Nipype WorkFlow
        """
        # Infosource Node - To iterate on subject and runs
        infosource_3rdlevel = Node(
            IdentityInterface(
                fields=[
                    "contrast_id",
                    "exp_dir",
                    "result_dir",
                    "output_dir",
                    "working_dir",
                    "subject_list",
                    "method",
                ],
                exp_dir=self.directories.dataset_dir,
                result_dir=self.directories.results_dir,
                output_dir=self.directories.output_dir,
                working_dir=self.directories.working_dir,
                subject_list=self.subject_list,
                method=method,
            ),
            name="infosource_3rdlevel",
        )
        infosource_3rdlevel.iterables = [("contrast_id", self.contrast_list)]

        # Templates to select files node
        copes_file = join(
            self.directories.output_dir,
            "l2_analysis",
            "_contrast_id_{contrast_id}_subject_id_*",
            "cope1.nii.gz",
        )

        varcopes_file = join(
            self.directories.output_dir,
            "l2_analysis",
            "_contrast_id_{contrast_id}_subject_id_*",
            "varcope1.nii.gz",
        )

        participants_file = join(self.directories.dataset_dir, "participants.tsv")

        mask_file = join(
            self.directories.dataset_dir, "..", "NARPS-X19V", "hypo1_unthresh.nii.gz"
        )

        template = {
            "cope": copes_file,
            "varcope": varcopes_file,
            "participants": participants_file,
            "mask": mask_file,
        }

        # SelectFiles node - to select necessary files
        selectfiles_3rdlevel = Node(
            SelectFiles(template, base_directory=str(self.directories.results_dir)),
            name="selectfiles_3rdlevel",
        )

        datasink_3rdlevel = Node(
            DataSink(
                base_directory=str(self.directories.results_dir),
                container=self.directories.output_dir,
            ),
            name="datasink_3rdlevel",
        )

        merge_copes_3rdlevel = Node(Merge(dimension="t"), name="merge_copes_3rdlevel")
        merge_varcopes_3rdlevel = Node(
            Merge(dimension="t"), name="merge_varcopes_3rdlevel"
        )

        subgroups_contrasts = Node(
            Function(
                input_names=["copes", "varcopes", "subject_ids", "participants_file"],
                output_names=[
                    "copes_equalIndifference",
                    "copes_equalRange",
                    "copes_global",
                    "varcopes_equalIndifference",
                    "varcopes_equalRange",
                    "varcopes_global",
                    "equalIndifference_id",
                    "equalRange_id",
                ],
                function=self.get_subgroups_contrasts,
            ),
            name="subgroups_contrasts",
        )

        specifymodel_3rdlevel = Node(
            MultipleRegressDesign(), name="specifymodel_3rdlevel"
        )

        flame_3rdlevel = Node(FLAMEO(run_mode="flame1"), name="flame_3rdlevel")

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
        regs.inputs.subject_list = self.subject_list

        smoothest = MapNode(
            SmoothEstimate(), name="smoothest", iterfield=["zstat_file"]
        )

        cluster = MapNode(
            Cluster(
                threshold=2.3,
                out_threshold_file=True,
                out_pval_file=True,
                pthreshold=0.05,
            ),
            name="cluster",
            iterfield=["in_file", "dlh", "volume", "cope_file"],
            synchronize=True,
        )

        n_sub = len(self.subject_list)
        l3_analysis = Workflow(
            base_dir=join(self.directories.results_dir, self.directories.working_dir),
            name=f"l3_analysis_{method}_nsub_{n_sub}",
        )

        l3_analysis.connect(
            [
                (
                    infosource_3rdlevel,
                    selectfiles_3rdlevel,
                    [("contrast_id", "contrast_id")],
                ),
                (
                    infosource_3rdlevel,
                    subgroups_contrasts,
                    [("subject_list", "subject_ids")],
                ),
                (
                    selectfiles_3rdlevel,
                    subgroups_contrasts,
                    [
                        ("cope", "copes"),
                        ("varcope", "varcopes"),
                        ("participants", "participants_file"),
                    ],
                ),
                (selectfiles_3rdlevel, flame_3rdlevel, [("mask", "mask_file")]),
                (selectfiles_3rdlevel, smoothest, [("mask", "mask_file")]),
                (
                    subgroups_contrasts,
                    regs,
                    [
                        ("equalRange_id", "equalRange_id"),
                        ("equalIndifference_id", "equalIndifference_id"),
                    ],
                ),
                (regs, specifymodel_3rdlevel, [("regressors", "regressors")]),
            ]
        )

        if method == "equalIndifference":
            specifymodel_3rdlevel.inputs.contrasts = [
                ["group_mean", "T", ["group_mean"], [1]],
                ["group_mean_neg", "T", ["group_mean"], [-1]],
            ]

            l3_analysis.connect(
                [
                    (
                        subgroups_contrasts,
                        merge_copes_3rdlevel,
                        [("copes_equalIndifference", "in_files")],
                    ),
                    (
                        subgroups_contrasts,
                        merge_varcopes_3rdlevel,
                        [("varcopes_equalIndifference", "in_files")],
                    ),
                ]
            )
        elif method == "equalRange":
            specifymodel_3rdlevel.inputs.contrasts = [
                ["group_mean", "T", ["group_mean"], [1]],
                ["group_mean_neg", "T", ["group_mean"], [-1]],
            ]

            l3_analysis.connect(
                [
                    (
                        subgroups_contrasts,
                        merge_copes_3rdlevel,
                        [("copes_equalRange", "in_files")],
                    ),
                    (
                        subgroups_contrasts,
                        merge_varcopes_3rdlevel,
                        [("varcopes_equalRange", "in_files")],
                    ),
                ]
            )

        elif method == "groupComp":
            specifymodel_3rdlevel.inputs.contrasts = [
                ["equalRange_sup", "T", ["equalRange", "equalIndifference"], [1, -1]]
            ]

            l3_analysis.connect(
                [
                    (
                        subgroups_contrasts,
                        merge_copes_3rdlevel,
                        [("copes_global", "in_files")],
                    ),
                    (
                        subgroups_contrasts,
                        merge_varcopes_3rdlevel,
                        [("varcopes_global", "in_files")],
                    ),
                ]
            )

        l3_analysis.connect(
            [
                (merge_copes_3rdlevel, flame_3rdlevel, [("merged_file", "cope_file")]),
                (
                    merge_varcopes_3rdlevel,
                    flame_3rdlevel,
                    [("merged_file", "var_cope_file")],
                ),
                (
                    specifymodel_3rdlevel,
                    flame_3rdlevel,
                    [
                        ("design_mat", "design_file"),
                        ("design_con", "t_con_file"),
                        ("design_grp", "cov_split_file"),
                    ],
                ),
                (
                    flame_3rdlevel,
                    cluster,
                    [("zstats", "in_file"), ("copes", "cope_file")],
                ),
                (flame_3rdlevel, smoothest, [("zstats", "zstat_file")]),
                (smoothest, cluster, [("dlh", "dlh"), ("volume", "volume")]),
                (
                    flame_3rdlevel,
                    datasink_3rdlevel,
                    [
                        ("zstats", f"l3_analysis_{method}_nsub_{n_sub}.@zstats"),
                        ("tstats", f"l3_analysis_{method}_nsub_{n_sub}.@tstats"),
                    ],
                ),
                (
                    cluster,
                    datasink_3rdlevel,
                    [
                        (
                            "threshold_file",
                            f"l3_analysis_{method}_nsub_{n_sub}.@thresh",
                        ),
                        ("pval_file", f"l3_analysis_{method}_nsub_{n_sub}.@pval"),
                    ],
                ),
            ]
        )

        return l3_analysis

    def get_hypotheses_outputs():
        """Get output for each hypothesis.

        Not yet implemented.
        """
        ...

    def reorganize_results(self):
        """Reorganize the results to analyze them.

        TODO must match behavior of get_hypotheses_outputs in parent class
        """
        n_sub = len(self.subject_list)
        tmp = [
            ("equalIndifference", 1),
            ("equalRange", 1),
            ("equalIndifference", 1),
            ("equalRange", 1),
            ("equalIndifference", 2),
            ("equalRange", 2),
            ("equalIndifference", 2),
            ("equalRange", 2),
            ("groupComp", 2),
        ]
        h = [
            join(
                self.directories.results_dir,
                self.directories.output_dir,
                f"l3_analysis_{grp}_nsub_{n_sub}",
                f"_contrast_id_{contrast}",
            )
            for grp, contrast in tmp
        ]

        repro_unthresh = [
            join(filename, "zstat1.nii.gz")
            if i in [4, 5]
            else join(filename, "zstat1.nii.gz")
            for i, filename in enumerate(h)
        ]

        repro_thresh = [
            join(filename, "_cluster0", "zstat1_threshold.nii.gz")
            if i in [4, 5]
            else join(filename, "_cluster0", "zstat1_threshold.nii.gz")
            for i, filename in enumerate(h)
        ]

        if not os.path.isdir(join(self.directories.results_dir, "NARPS-reproduction")):
            os.mkdir(join(self.directories.results_dir, "NARPS-reproduction"))

        for i, filename in enumerate(repro_unthresh):
            f_in = filename
            f_out = join(
                self.directories.results_dir,
                "NARPS-reproduction",
                f"team_{self.team_id}_nsub_{n_sub}_hypo{i+1}_unthresholded.nii.gz",
            )
            shutil.copyfile(f_in, f_out)

        for i, filename in enumerate(repro_thresh):
            f_in = filename
            img = nb.load(filename)
            original_affine = img.affine.copy()
            spm = nb.load(repro_unthresh[i])
            new_img = img.get_fdata() > 0.95
            new_img = new_img.astype(float) * spm.get_fdata()
            new_spm = nb.Nifti1Image(new_img, original_affine)
            nb.save(
                new_spm,
                join(
                    self.directories.results_dir,
                    "NARPS-reproduction",
                    f"team_{self.team_id}_nsub_{n_sub}_hypo{i+1}_thresholded.nii.gz",
                ),
            )
            # f_out = join(result_dir, "final_results", f"team_{team_ID}_nsub_{n_sub}_hypo{i+1}_thresholded.nii.gz")
            # shutil.copyfile(f_in, f_out)

        print(f"Results files of team {self.team_id} reorganized.")

        return h


def get_subject_infos(event_file: str):
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
        onset[c] = []
        duration[c] = []
        amplitude[c] = []

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

    subject_info = [
        Bunch(
            conditions=cond_names,
            onsets=[onset[k] for k in cond_names],
            durations=[duration[k] for k in cond_names],
            amplitudes=[amplitude[k] for k in cond_names],
            regressor_names=None,
            regressors=None,
        )
    ]

    return subject_info


# [INFO] Linear contrast effects: 'Gain' vs. baseline, 'Loss' vs. baseline.
def get_contrasts() -> list[tuple]:
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
