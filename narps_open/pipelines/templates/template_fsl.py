#!/usr/bin/python
# coding: utf-8

"""
This template can be use to reproduce a pipeline using FSL as main software.

- Replace all occurrences of 48CD by the actual id of the team.
- All lines starting with [INFO], are meant to help you during the reproduction, these can be removed
eventually.
- Also remove lines starting with [TODO], once you did what they suggested.
"""

# [TODO] Only import modules you use further in te code, remove others from the import section

from os.path import join

# [INFO] The import of base objects from Nipype, to create Workflows
from nipype import Node, Workflow # , JoinNode, MapNode

# [INFO] a list of interfaces used to manpulate data
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
# from nipype.algorithms.misc import Gunzip

# [INFO] a list of FSL-specific interfaces
# from nipype.algorithms.modelgen import SpecifyModel
# from nipype.interfaces.fsl import (
#     Info, ImageMaths, IsotropicSmooth, Threshold,
#     Level1Design, FEATModel, L2Model, Merge,
#     FLAMEO, ContrastMgr, FILMGLS, MultipleRegressDesign,
#     Cluster, BET, SmoothEstimate
#     )

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
        """ Return a Nipype workflow describing the prerpocessing part of the pipeline """

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
        # at the beginning of the file.
        node_name = Node(
            node_function,
            name='node_name'
        )

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

    # [INFO] There was no run level analysis for the pipelines using FSL
    def get_run_level_analysis(self):
        """ Return a Nipype workflow describing the run level analysis part of the pipeline """
        return None

    # [INFO] This function is used in the subject level analysis pipelines using FSL
    # [TODO] Adapt this example to your specific pipeline
    def get_session_infos(event_file: str):
        """
        Create Bunchs for specifyModel.

        Parameters :
        - event_file : file corresponding to the run and the subject to analyze

        Returns :
        - subject_info : list of Bunch for 1st level analysis.
        """

        condition_names = ['trial', 'gain', 'loss']

        onset = {}
        duration = {}
        amplitude = {}

        # Creates dictionary items with empty lists for each condition.
        for condition in condition_names:  
            onset.update({condition: []}) 
            duration.update({condition: []})
            amplitude.update({condition: []})

        with open(event_file, 'rt') as file:
            next(file)  # skip the header

            for line in file:
                info = line.strip().split()
                # Creates list with onsets, duration and loss/gain for amplitude (FSL)
                for condition in condition_names:
                    if condition == 'gain':
                        onset[condition].append(float(info[0]))
                        duration[condition].append(float(info[4]))
                        amplitude[condition].append(float(info[2]))
                    elif condition == 'loss':
                        onset[condition].append(float(info[0]))
                        duration[condition].append(float(info[4]))
                        amplitude[condition].append(float(info[3]))
                    elif condition == 'trial':
                        onset[condition].append(float(info[0]))
                        duration[condition].append(float(info[4]))
                        amplitude[condition].append(float(1))

        subject_info = []
        subject_info.append(
            Bunch(
                conditions = condition_names,
                onsets = [onset[k] for k in condition_names],
                durations = [duration[k] for k in condition_names],
                amplitudes = [amplitude[k] for k in condition_names],
                regressor_names = None,
                regressors = None,
            )
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
        """ Return a Nipype workflow describing the subject level analysis part of the pipeline """

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
        # at the beginning of the file.
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
    def get_subgroups_contrasts(
        copes, varcopes, subject_list: list, participants_file: str
    ):
        """
        This function return the file list containing only the files
        belonging to subject in the wanted group.

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
        with open(participants_file, 'rt') as file:
            next(file)  # skip the header

            for line in file:
                info = line.strip().split()

                # Checking for each participant if its ID was selected
                # and separate people depending on their group
                if info[0][-3:] in subject_list and info[1] == 'equalIndifference':
                    equal_indifference_id.append(info[0][-3:])
                elif info[0][-3:] in subject_list and info[1] == 'equalRange':
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
            sub_id = cope.split('/')
            if sub_id[-2][-3:] in equal_indifference_id:
                copes_equal_indifference.append(cope)
            elif sub_id[-2][-3:] in equal_range_id:
                copes_equal_range.append(cope)
            if sub_id[-2][-3:] in subject_list:
                copes_global.append(cope)

            sub_id = varcope.split('/')
            if sub_id[-2][-3:] in equal_indifference_id:
                varcopes_equal_indifference.append(varcope)
            elif sub_id[-2][-3:] in equal_range_id:
                varcopes_equal_range.append(varcope)
            if sub_id[-2][-3:] in subject_list:
                varcopes_global.append(varcope)

        return copes_equal_indifference, copes_equal_range,
            varcopes_equal_indifference, varcopes_equal_range,
            equal_indifference_id, equal_range_id,
            copes_global, varcopes_global


    # [INFO] This function creates the dictionary of regressors used in FSL Nipype pipelines
    def get_regressors(
        equal_range_id: list,
        equal_indifference_id: list,
        method: str,
        subject_list: list,
    ) -> dict:
        """
        Create dictionary of regressors for group analysis.

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
        if method == 'equalRange':
            regressors = dict(group_mean = [1 for i in range(len(equal_range_id))])
        elif method == 'equalIndifference':
            regressors = dict(group_mean = [1 for i in range(len(equal_indifference_id))])

        # For two sample t-test, creates 2 lists:
        #  - one for equal range group,
        #  - one for equal indifference group
        # Each list contains n_sub values with 0 and 1 depending on the group of the participant
        # For equalRange_reg list --> participants with a 1 are in the equal range group
        elif method == 'groupComp':
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
                equalRange = equalRange_reg,
                equalIndifference = equalIndifference_reg
            )

        return regressors

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
        template = {
            'cope' : join(
                self.directories.results_dir,
                'subject_level_analysis',
                '_contrast_id_{contrast_id}_subject_id_*', 'cope1.nii.gz'),
            'varcope' : join(
                self.directories.results_dir,
                'subject_level_analysis',
                '_contrast_id_{contrast_id}_subject_id_*', 'varcope1.nii.gz'),
            'participants' : join(
                self.directories.dataset_dir,
                'participants.tsv')
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

        contrasts = Node(
            Function(
                input_names=['copes', 'varcopes', 'subject_ids', 'participants_file'],
                output_names=[
                    'copes_equalIndifference',
                    'copes_equalRange',
                    'varcopes_equalIndifference',
                    'varcopes_equalRange',
                    'equalIndifference_id',
                    'equalRange_id',
                    'copes_global',
                    'varcopes_global'
                ],
                function = self.get_subgroups_contrasts,
            ),
            name = 'subgroups_contrasts',
        )

        regs = Node(
            Function(
                input_names = [
                    'equalRange_id',
                    'equalIndifference_id',
                    'method',
                    'subject_list',
                ],
                output_names = ['regressors'],
                function = self.get_regressors,
            ),
            name = 'regs',
        )
        regs.inputs.method = method
        regs.inputs.subject_list = subject_list

        # [INFO] The following part has to be modified with nodes of the pipeline

        # [TODO] For each node, replace 'node_name' by an explicit name, and use it for both:
        #   - the name of the variable in which you store the Node object
        #   - the 'name' attribute of the Node
        # [TODO] The node_function refers to a NiPype interface that you must import
        # at the beginning of the file.
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
                (
                    info_source,
                    subgroups_contrasts,
                    [('subject_list', 'subject_ids')],
                ),
                (
                    select_files,
                    subgroups_contrasts,
                    [
                        ('cope', 'copes'),
                        ('varcope', 'varcopes'),
                        ('participants', 'participants_file'),
                    ],
                ),
                (
                    select_files,
                    node_name[('func', 'node_input_name')],
                ),
                (
                    node_variable,
                    datasink_groupanalysis,
                    [('node_output_name', 'preprocess.@sym_link')],
                ),
            ]
        ) # Complete with other links between nodes

        # [INFO] Here we define the contrasts used for the group level analysis, depending on the
        # method used.
        if method in ('equalRange', 'equalIndifference'):
            contrasts = [('Group', 'T', ['mean'], [1]), ('Group', 'T', ['mean'], [-1])]

        elif method == 'groupComp':
            contrasts = [
                ('Eq range vs Eq indiff in loss', 'T', ['Group_{1}', 'Group_{2}'], [1, -1])
            ]

        # [INFO] Here we simply return the created workflow
        return group_level_analysis
