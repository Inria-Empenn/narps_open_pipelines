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

        # [INFO] You may for example define the contrasts that will be analyzed
        # in the run level analysis. Each contrast is in the form :
        # [Name, Stat, [list of condition names], [weights on those conditions]]
        self.run_level_contrasts = [
            ['trial', 'T', ['trial', 'trialxgain^1', 'trialxloss^1'], [1, 0, 0]],
            ['effect_of_gain', 'T', ['trial', 'trialxgain^1', 'trialxloss^1'], [0, 1, 0]],
            ['effect_of_loss', 'T', ['trial', 'trialxgain^1', 'trialxloss^1'], [0, 0, 1]]
        ]

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

        # SelectFiles node - to select necessary files
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
        select_files = Node(SelectFiles(file_templates), name='select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir

        # DataSink Node - store the wanted results in the wanted repository
        data_sink = Node(DataSink(), name='data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir

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
        preprocessing.connect([
            (info_source, select_files, [('subject_id', 'subject_id'), ('run_id', 'run_id')]),
            (select_files, node_name, [('func', 'node_input_name')]),
            (node_name, data_sink, [('node_output_name', 'preprocessing.@sym_link')])
        ])

        # [INFO] Here we simply return the created workflow
        return preprocessing

    # [INFO] This function is used in the run level analysis, in order to 
    #   extract trial information from the event files
    # [TODO] Adapt this example to your specific pipeline
    def get_subject_information(event_file: str):
        """
        Extract information from an event file, to setup the model.

        Parameters :
        - event_file : str, event file corresponding to the run and the subject to analyze

        Returns :
        - subject_info : list of Bunch containing event information
        """
        # [INFO] nipype requires to import all dependencies from inside the methods that are
        #   later used in Function nodes
        from nipype.interfaces.base import Bunch

        condition_names = ['event', 'gain', 'loss', 'response']
        onsets = {}
        durations = {}
        amplitudes = {}

        # Create dictionary items with empty lists
        for condition in condition_names:
            onsets.update({condition : []})
            durations.update({condition : []})
            amplitudes.update({condition : []})

        # Parse information in the event_file
        with open(event_file, 'rt') as file:
            next(file)  # skip the header

            for line in file:
                info = line.strip().split()
                onsets['event'].append(float(info[0]))
                durations['event'].append(float(info[1]))
                amplitudes['event'].append(1.0)
                onsets['gain'].append(float(info[0]))
                durations['gain'].append(float(info[1]))
                amplitudes['gain'].append(float(info[2]))
                onsets['loss'].append(float(info[0]))
                durations['loss'].append(float(info[1]))
                amplitudes['loss'].append(float(info[3]))
                onsets['response'].append(float(info[0]))
                durations['response'].append(float(info[1]))
                if 'accept' in info[5]:
                    amplitudes['response'].append(1.0)
                elif 'reject' in info[5]:
                    amplitudes['response'].append(-1.0)
                else:
                    amplitudes['response'].append(0.0)

        return [
            Bunch(
                conditions = condition_names,
                onsets = [onsets[k] for k in condition_names],
                durations = [durations[k] for k in condition_names],
                amplitudes = [amplitudes[k] for k in condition_names],
                regressor_names = None,
                regressors = None)
            ]

    def get_run_level_analysis(self):
        """ Return a Nipype workflow describing the run level analysis part of the pipeline """

        # [INFO] The following part stays the same for all pipelines
        # [TODO] Modify the templates dictionary to select the files
        #    that are relevant for your analysis only.

        # IdentityInterface node - allows to iterate over subjects and runs
        information_source = Node(IdentityInterface(
            fields = ['subject_id', 'run_id']),
            name = 'information_source')
        information_source.iterables = [
            ('run_id', self.run_list),
            ('subject_id', self.subject_list),
        ]

        # SelectFiles node - to select necessary files
        templates = {
            # Functional MRI - computed by preprocessing
            'func' : join(self.directories.output_dir, 'preprocessing',
                '_run_id_{run_id}_subject_id_{subject_id}',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_brain_mcf_st_smooth_flirt_wtsimt.nii.gz'
                ),
            # Event file - from the original dataset
            'event' : join('sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_events.tsv'
                ),
            # Motion parameters - computed by preprocessing's motion_correction Node
            'motion' : join(self.directories.output_dir, 'preprocessing',
                '_run_id_{run_id}_subject_id_{subject_id}',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_brain_mcf.nii.gz.par',
                )
        }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir

        # DataSink Node - store the wanted results in the wanted directory
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir

        # [TODO] Continue adding nodes to the run level analysis part of the pipeline

        # [INFO] The following part defines the nipype workflow and the connections between nodes
        run_level_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = 'run_level_analysis'
        )

        # [TODO] Add the connections the workflow needs
        # [INFO] Input and output names can be found on NiPype documentation
        run_level_analysis.connect([
            (info_source, select_files, [('subject_id', 'subject_id'), ('run_id', 'run_id')])
            # [TODO] Add other connections here
        ])

        # [INFO] Here we simply return the created workflow
        return run_level_analysis

    def get_subject_level_analysis(self):
        """ Return a Nipype workflow describing the subject level analysis part of the pipeline """

        # [INFO] The following part stays the same for all pipelines

        # [TODO] Define a self.contrast_list in the __init__() method. It will allow to iterate
        #   on contrasts computed in the run level analysis

        # Infosource Node - To iterate on subjects
        info_source = Node(IdentityInterface(
                fields = ['subject_id', 'contrast_id']),
            name='info_source')
        information_source.iterables = [
            ('subject_id', self.subject_list),
            ('contrast_id', self.contrast_list)
            ]

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
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir

        # DataSink Node - store the wanted results in the wanted repository
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir

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
            (info_source, select_files, [('subject_id', 'subject_id')]),
            (info_source, contrasts, [('subject_id', 'subject_id')]),
            (select_files, subject_infos, [('event', 'event_files')]),
            (select_files, node_name, [('func', 'node_input_name')]),
            (node_name, data_sink, [('node_output_name', 'preprocess.@sym_link')])
        ])

        # [INFO] Here we simply return the created workflow
        return subject_level_analysis

    # [INFO] This function creates the dictionary of regressors used in FSL Nipype pipelines
    def get_one_sample_t_test_regressors(subject_list: list) -> dict:
        """
        Create dictionary of regressors for one sample t-test group analysis.

        Parameters:
            - subject_list: ids of subject in the group for which to do the analysis

        Returns:
            - dict containing named lists of regressors.
        """

        return dict(group_mean = [1 for _ in subject_list])

    # [INFO] This function creates the dictionary of regressors used in FSL Nipype pipelines
    def get_two_sample_t_test_regressors(
        equal_range_ids: list,
        equal_indifference_ids: list,
        subject_list: list,
        ) -> dict:
        """
        Create dictionary of regressors for two sample t-test group analysis.

        Parameters:
            - equal_range_ids: ids of subjects in equal range group
            - equal_indifference_ids: ids of subjects in equal indifference group
            - subject_list: ids of subject for which to do the analysis

        Returns:
            - regressors, dict: containing named lists of regressors.
            - groups, list: group identifiers to distinguish groups in FSL analysis.
        """

        # Create 2 lists containing n_sub values which are
        #  * 1 if the participant is on the group
        #  * 0 otherwise
        equal_range_regressors = [1 if i in equal_range_ids else 0 for i in subject_list]
        equal_indifference_regressors = [
            1 if i in equal_indifference_ids else 0 for i in subject_list
            ]

        # Create regressors output : a dict with the two list
        regressors = dict(
            equalRange = equal_range_regressors,
            equalIndifference = equal_indifference_regressors
        )

        # Create groups outputs : a list with 1 for equalRange subjects and 2 for equalIndifference
        groups = [1 if i == 1 else 2 for i in equal_range_regressors]

        return regressors, groups
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
        information_source = Node(
            IdentityInterface(
                fields = ['contrast_id']
            ),
            name = 'information_source',
        )
        information_source.iterables = [('contrast_id', self.contrast_list)]

        # Templates to select files node
        # [TODO] Change the name of the files depending on the filenames
        # of results of first level analysis
        templates = {
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
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.results_dir,
        select_files.inputs.force_list = True

        # Datasink node - to save important files
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir

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
        group_level_analysis.connect([
            (info_source, select_files, [('contrast_id', 'contrast_id')]),
            (info_source, subgroups_contrasts, [('subject_list', 'subject_ids')]),
            (select_files, subgroups_contrasts,[
                        ('cope', 'copes'),
                        ('varcope', 'varcopes'),
                        ('participants', 'participants_file'),
                    ]),
            (select_files, node_name[('func', 'node_input_name')]),
            (node_variable, datasink_groupanalysis,
                [('node_output_name', 'preprocess.@sym_link')])
        ]) # Complete with other links between nodes

        # [INFO] You can add conditional sections of code to shape the workflow depending
        #   on the method passed as parameter
        if method in ('equalRange', 'equalIndifference'):
            contrasts = [('Group', 'T', ['mean'], [1]), ('Group', 'T', ['mean'], [-1])]

        elif method == 'groupComp':
            contrasts = [
                ('Eq range vs Eq indiff in loss', 'T', ['Group_{1}', 'Group_{2}'], [1, -1])
            ]

        # [INFO] Here we simply return the created workflow
        return group_level_analysis
