#!/usr/bin/python
# coding: utf-8

"""
This template can be use to reproduce a pipeline using SPM as main software.

- Replace all occurences of 48CD by the actual id of the team.
- All lines beging [INFO], are meant to help you during the reproduction, these can be removed
eventually.
- Also remove lines beging with [TODO], once you did what they suggested.
"""

# [TODO] Only import modules you use further in te code, remove others from the import section

from os.path import join

# [INFO] The import of base objects from Nipype, to create Workflows
from nipype import Node, Workflow # , JoinNode, MapNode

# [INFO] a list of interfaces used to manpulate data
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
# from nipype.algorithms.misc import Gunzip

# [INFO] a list of SPM-specific interfaces
# from nipype.algorithms.modelgen import SpecifySPMModel
# from nipype.interfaces.spm import (
#    Smooth,
#    Level1Design, OneSampleTTestDesign, TwoSampleTTestDesign,
#    EstimateModel, EstimateContrast, Threshold
#    )

# [INFO] In order to inherit from Pipeline
from narps_open.pipelines import Pipeline

class PipelineTeam48CD(Pipeline):
    """ A class that defines the pipeline of team 48CD """

    def __init__(self):
        # [INFO] Remove the init method completely if unused
        # [TODO] Init the attributes of the pipeline, if any other than the ones defined
        # in the pipeline class
        pass

    def get_preprocessing(self):
        """ Return a Nipype worflow describing the prerpocessing part of the pipeline """

        # [INFO] The following part stays the same for all preprocessing pipelines

        # IdentityInterface node - allows to iterate over subjects and runs
        info_source = Node(
            IdentityInterface(fields=['subject_id', 'run_id']),
            name='info_source'
        )
        info_source.iterables = [
            ('subject_id', self.subject_list),
            ('run_id', self.run_list),
        ]

        # Templates to select files node
        file_templates = {
            'anat': join(
                'sub-{subject_id}', 'anat', 'sub-{subject_id}_T1w.nii.gz'
                ),
            'func': join(
                'sub-{subject_id}', 'func', 'sub-{subject_id}_task-MGT_run-{run_id}_bold.nii.gz'
                ),
            'magnitude': join(
                'sub-{subject_id}', 'fmap', 'sub-{subject_id}_magnitude1.nii.gz'
                ),
            'phasediff': join(
                'sub-{subject_id}', 'fmap', 'sub-{subject_id}_phasediff.nii.gz'
                )
        }

        # SelectFiles node - to select necessary files
        select_files = Node(
            SelectFiles(file_templates, base_directory = self.directories.dataset_dir),
            name='select_files'
        )

        # DataSink Node - store the wanted results in the wanted repository
        data_sink = Node(
            DataSink(base_directory = self.directories.output_dir),
            name='data_sink',
        )

        # [INFO] The following part has to be modified with nodes of the pipeline

        # [TODO] For each node, replace 'node_name' by an explicit name, and use it for both:
        #   - the name of the variable in which you store the Node object
        #   - the 'name' attribute of the Node
        # [TODO] The node_function refers to a NiPype interface that you must import
        # at the begining of the file.
        node_name = Node(
            node_function,
            name='node_name'
        )

# --- For each run

# 'preprocessing.motion_correction': 'SPM12, Realign & Unwarp using the Phase map created with SPM12 Fieldmap Toolbox v2.1 (default options): Other than defaults: Estimation: Quality 0.95, Speparation 3, Register to mean, Interpolation 7th Degree B-Spline; Unwarp Reslice: Interpolation 7th Degree B-Spline 

# We reuse the same total readout time, blip direction and echo times as for team xxxx as this is the same dataset (NARPS)

        
fieldmap = Node(FieldMap(), name = 'fieldmap')

# madatory
        blip_direction = -1
        echo_times = (4.92, 7.38)
        epi_file = '_bold.nii.gz', '_sbref.nii.gz'
        magnitude_file = magnitude1.nii.gz
        phase_file = phase_diff.nii.gz
        total_readout_time = 29.15
# optionnal
        quality = 0.95
        separation = 3
        register_to_mean = True
        interp = 7
        reslice_interp = 7

# ---> outputs: mean_image 
# modified_in_files{1} : distortion corrected single-band reference EPI
# modified_in_files{2} : distortion corrected EPI time-series

# 'preprocessing.inter_subject_reg': 'SPM12: Within each run, the distortion corrected single-band reference EPI was co-registered to the mean EPI from Realignment & Warp using normalised mutual information.


coreg1 = spm.Coregister()
coreg1.inputs.target = fieldmap.mean_image # output of realign unwarp
coreg1.inputs.source = fieldmap.modified_in_files{1} # distortion corrected single-band reference EPI 
coreg1.cost_function = 'nmi'
# ---> output : coregistered_files : distortion corrected single-band reference EPI coreg onto mean image

# Then distortion corrected single-band reference EPI was co-registered to the gray matter probability map in the Old Segmentation toolbox in SPM using normalised mutual information and the distortion corrected EPI time-series as well as the mean EPI remained aligned. 

# segment the structural data to get the grey matter
seg = spm.Segment()
seg.inputs.data = 'structural.nii'
# --> output: 
# native_gm_image
# transformation_mat 

coreg2 = spm.Coregister()
coreg2.inputs.target = seg.native_gm_image
coreg2.inputs.source = coreg1.coregistered_files # distortion corrected single-band reference EPI coreg onto mean image 
coreg2.cost_function = 'nmi'
coreg2.others = {
    fieldmap.modified_in_files{2} #'distortion corrected EPI time-series', 
    fieldmap.mean_image}
# ---> output : coregistered_files 

# The single-band reference EPI was normalized to the SPM MNI152 template space using the classic Unified Segmentation approach in the Old Segment function in SPM, while mitigating overfitting by setting the warp frequency cutoff to 45 limiting the discrete cosine transform (DCT) bases and setting the sampling distance to 2.

norm = spm.Normalize()
norm.inputs.parameter_file = seg.transformation_mat
norm.inputs.source = coreg2.coregistered_files







        # [TODO] Add another node for each step of the pipeline

        # [INFO] The following part defines the nipype workflow and the connections between nodes

        preprocessing = Workflow(
            base_dir = self.directories.working_dir,
            name = 'preprocessing'
        )

        # [TODO] Add the connections the workflow needs
        # [INFO] Input and output names can be found on NiPype documentation
        preprocessing.connect(
            [
                (
                    info_source,
                    select_files,
                    [('subject_id', 'subject_id'), ('run_id', 'run_id')],
                ),
                (
                    select_files,
                    node_name,
                    [('func', 'node_input_name')],
                ),
                (
                    node_name,
                    data_sink,
                    [('node_output_name', 'preprocessing.@sym_link')],
                ),
            ]
        )

        # [INFO] Here we simply return the created workflow
        return preprocessing

    # [INFO] There was no run level analysis for the pipelines using SPM
    def get_run_level_analysis(self):
        """ Return a Nipype worflow describing the run level analysis part of the pipeline """
        return None

    # [INFO] This function is used in the subject level analysis pipelines using SPM
    # [TODO] Adapt this example to your specific pipeline
    def get_subject_infos(event_files: list, runs: list):
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

        condition_names = ['trial', 'accepting', 'rejecting']
        onset = {}
        duration = {}
        weights_gain = {}
        weights_loss = {}
        onset_button = {}
        duration_button = {}

        # Loop over number of runs
        for run_id in range(len(runs)):

            # Create dictionary items with empty lists
            onset.update({s + '_run' + str(run_id + 1): [] for s in condition_names})
            duration.update({s + '_run' + str(run_id + 1): [] for s in condition_names})
            weights_gain.update({'gain_run' + str(run_id + 1): []})
            weights_loss.update({'loss_run' + str(run_id + 1): []})

            with open(event_files[run_id], 'rt') as event_file:
                next(event_file)  # skip the header

                for line in event_file:
                    info = line.strip().split()

                    for condition in condition_names:
                        val = condition + '_run' + str(run_id + 1)  # trial_run1 or accepting_run1
                        val_gain = 'gain_run' + str(run_id + 1)  # gain_run1
                        val_loss = 'loss_run' + str(run_id + 1)  # loss_run1
                        if condition == 'trial':
                            onset[val].append(float(info[0]))  # onsets for trial_run1
                            duration[val].append(float(4))
                            weights_gain[val_gain].append(float(info[2]))
                            weights_loss[val_loss].append(float(info[3]))
                        elif condition == 'accepting' and 'accept' in info[5]:
                            onset[val].append(float(info[0]) + float(info[4]))
                            duration[val].append(float(0))
                        elif condition == 'rejecting' and 'reject' in info[5]:
                            onset[val].append(float(info[0]) + float(info[4]))
                            duration[val].append(float(0))

        # Bunching is done per run, i.e. trial_run1, trial_run2, etc.
        # But names must not have '_run1' etc because we concatenate runs
        subject_info = []
        for run_id in range(len(runs)):

            conditions = [s + '_run' + str(run_id + 1) for s in condition_names]
            gain = 'gain_run' + str(run_id + 1)
            loss = 'loss_run' + str(run_id + 1)

            subject_info.insert(
                run_id,
                Bunch(
                    conditions = condition_names,
                    onsets = [onset[c] for c in conditions],
                    durations = [duration[c] for c in conditions],
                    amplitudes = None,
                    tmod = None,
                    pmod = [
                        Bunch(
                            name = ['gain', 'loss'],
                            poly = [1, 1],
                            param = [weights_gain[gain], weights_loss[loss]],
                        ),
                        None,
                    ],
                    regressor_names = None,
                    regressors = None,
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
        conditions = ['trial', 'trialxgain^1', 'trialxloss^1']

        # Create contrasts
        trial = ('trial', 'T', conditions, [1, 0, 0])
        effect_gain = ('effect_of_gain', 'T', conditions, [0, 1, 0])
        effect_loss = ('effect_of_loss', 'T', conditions, [0, 0, 1])

        # Contrast list
        return [trial, effect_gain, effect_loss]

    def get_subject_level_analysis(self):
        """ Return a Nipype worflow describing the subject level analysis part of the pipeline """

        # [INFO] The following part stays the same for all pipelines

        # Infosource Node - To iterate on subjects
        info_source = Node(
            IdentityInterface(
                fields = ['subject_id', 'dataset_dir', 'results_dir', 'working_dir', 'run_list'],
                dataset_dir = self.directories.dataset_dir,
                results_dir = self.directories.results_dir,
                working_dir = self.directories.working_dir,
                run_list = self.run_list
            ),
            name='info_source',
        )
        info_source.iterables = [('subject_id', self.subject_list)]

        # Templates to select files node
        # [TODO] Change the name of the files depending on the filenames of results of preprocessing
        templates = {
            'func': join(
                self.directories.results_dir,
                'preprocess',
                '_run_id_*_subject_id_{subject_id}',
                'complete_filename_{subject_id}_complete_filename.nii',
            ),
            'event': join(
                self.directories.dataset_dir,
                'sub-{subject_id}',
                'func',
                'sub-{subject_id}_task-MGT_run-*_events.tsv',
            )
        }

        # SelectFiles node - to select necessary files
        select_files = Node(
            SelectFiles(templates, base_directory = self.directories.dataset_dir),
            name = 'select_files'
        )

        # DataSink Node - store the wanted results in the wanted repository
        data_sink = Node(
            DataSink(base_directory = self.directories.output_dir),
            name = 'data_sink'
        )

        # [INFO] This is the node executing the get_subject_infos_spm function
        # Subject Infos node - get subject specific condition information
        subject_infos = Node(
            Function(
                input_names = ['event_files', 'runs'],
                output_names = ['subject_info'],
                function = self.get_subject_infos,
            ),
            name = 'subject_infos',
        )
        subject_infos.inputs.runs = self.run_list

        # [INFO] This is the node executing the get_contrasts function
        # Contrasts node - to get contrasts
        contrasts = Node(
            Function(
                input_names = ['subject_id'],
                output_names = ['contrasts'],
                function = self.get_contrasts,
            ),
            name = 'contrasts',
        )

        # [INFO] The following part has to be modified with nodes of the pipeline

        # [TODO] For each node, replace 'node_name' by an explicit name, and use it for both:
        #   - the name of the variable in which you store the Node object
        #   - the 'name' attribute of the Node
        # [TODO] The node_function refers to a NiPype interface that you must import
        # at the begining of the file.
        node_name = Node(
            node_function,
            name = 'node_name'
        )

        # [TODO] Add other nodes with the different steps of the pipeline

        # [INFO] The following part defines the nipype workflow and the connections between nodes

        subject_level_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = 'subject_level_analysis'
        )
        # [TODO] Add the connections the workflow needs
        # [INFO] Input and output names can be found on NiPype documentation
        subject_level_analysis.connect([
            (
                info_source,
                select_files,
                [('subject_id', 'subject_id')]
            ),
            (
                info_source,
                contrasts,
                [('subject_id', 'subject_id')]
            ),
            (
                select_files,
                subject_infos,
                [('event', 'event_files')]
            ),
            (
                select_files,
                node_name,
                [('func', 'node_input_name')]
            ),
            (
                node_name, data_sink,
                [('node_output_name', 'preprocess.@sym_link')]
            ),
        ])

        # [INFO] Here we simply return the created workflow
        return subject_level_analysis

    # [INFO] This function returns the list of ids and files of each group of participants
    # to do analyses for both groups, and one between the two groups.
    def get_subset_contrasts(
        file_list, subject_list: list, participants_file: str
    ):
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
        with open(participants_file, 'rt') as file:
            next(file)  # skip the header
            for line in file:
                info = line.strip().split()
                if info[0][-3:] in subject_list and info[1] == 'equalIndifference':
                    equal_indifference_id.append(info[0][-3:])
                elif info[0][-3:] in subject_list and info[1] == 'equalRange':
                    equal_range_id.append(info[0][-3:])

        for file in file_list:
            sub_id = file.split('/')
            if sub_id[-2][-3:] in equal_indifference_id:
                equal_indifference_files.append(file)
            elif sub_id[-2][-3:] in equal_range_id:
                equal_range_files.append(file)

        return equal_indifference_id, equal_range_id, equal_indifference_files, equal_range_files

    def get_group_level_analysis(self):
        """
        Return all workflows for the group level analysis.

        Returns;
            - a list of nipype.WorkFlow
        """

        methods = ['equalRange', 'equalIndifference', 'groupComp']
        return [self.get_group_level_analysis_sub_workflow(method) for method in methods]

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
                fields = ['contrast_id', 'subjects'],
                subjects = self.subject_list
            ),
            name = 'info_source',
        )
        info_source.iterables = [('contrast_id', self.contrast_list)]

        # Templates to select files node
        # [TODO] Change the name of the files depending on the filenames
        # of results of first level analysis
        templates = {
            'contrast': join(
                self.directories.results_dir,
                'subject_level_analysis',
                '_subject_id_*',
                'complete_filename_{contrast_id}_complete_filename.nii',
            ),
            'participants': join(
                self.directories.dataset_dir,
                'participants.tsv'
            )
        }
        select_files = Node(
            SelectFiles(
                templates,
                base_directory = self.directories.results_dir,
                force_list = True
            ),
            name = 'select_files',
        )

        # Datasink node - to save important files
        data_sink = Node(
            DataSink(base_directory = self.directories.output_dir),
            name = 'data_sink',
        )

        # Contrasts node - to select subset of contrasts
        sub_contrasts = Node(
            Function(
                input_names = ['file_list', 'method', 'subject_list', 'participants_file'],
                output_names = [
                    'equalIndifference_id',
                    'equalRange_id',
                    'equalIndifference_files',
                    'equalRange_files',
                ],
                function = self.get_subset_contrasts,
            ),
            name = 'sub_contrasts',
        )
        sub_contrasts.inputs.method = method

        # [INFO] The following part has to be modified with nodes of the pipeline

        # [TODO] For each node, replace 'node_name' by an explicit name, and use it for both:
        #   - the name of the variable in which you store the Node object
        #   - the 'name' attribute of the Node
        # [TODO] The node_function refers to a NiPype interface that you must import
        # at the begining of the file.
        node_name = Node(
            node_function,
            name = 'node_name'
        )

        # [INFO] The following part defines the nipype workflow and the connections between nodes

        # Compute the number of participants used to do the analysis
        nb_subjects = len(self.subject_list)

        # Declare the workflow
        group_level_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = f'group_level_analysis_{method}_nsub_{nb_subjects}'
        )
        group_level_analysis.connect(
            [
                (
                    info_source,
                    select_files,
                    [('contrast_id', 'contrast_id')],
                ),
                (info_source, sub_contrasts, [('subjects', 'subject_list')]),
                (
                    select_files,
                    sub_contrasts,
                    [('contrast', 'file_list'), ('participants', 'participants_file')],
                ),  # Complete with other links between nodes
            ]
        )

        # [INFO] Here whe define the contrasts used for the group level analysis, depending on the
        # method used.
        if method in ('equalRange', 'equalIndifference'):
            contrasts = [('Group', 'T', ['mean'], [1]), ('Group', 'T', ['mean'], [-1])]

        elif method == 'groupComp':
            contrasts = [
                ('Eq range vs Eq indiff in loss', 'T', ['Group_{1}', 'Group_{2}'], [1, -1])
            ]

        # [INFO] Here we simply return the created workflow
        return group_level_analysis
