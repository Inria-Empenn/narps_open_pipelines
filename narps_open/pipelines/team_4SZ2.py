#!/usr/bin/python
# coding: utf-8

""" Write the work of NARPS team 4SZ2 using Nipype """

from os.path import join
from itertools import product

from numpy import array

from nipype import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Function, Split
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.fsl import (
    IsotropicSmooth, Level1Design, FEATModel,
    L2Model, Merge, FLAMEO, FILMGLS, MultipleRegressDesign,
    FSLCommand, Cluster
    )
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces.fsl.maths import MathsCommand

from narps_open.utils.configuration import Configuration
from narps_open.pipelines import Pipeline
from narps_open.data.task import TaskInformation
from narps_open.data.participants import get_group, get_participants_information
from narps_open.core.common import list_intersection, elements_in_string, clean_list
from narps_open.core.interfaces import InterfaceFactory

# Setup FSL
FSLCommand.set_default_output_type('NIFTI_GZ')

class PipelineTeam4SZ2(Pipeline):
    """ A class that defines the pipeline of team 4SZ2 """

    def __init__(self):
        super().__init__()
        self.fwhm = 5.0
        self.team_id = '4SZ2'
        self.contrast_list = ['1', '2']
        self.run_level_contrasts = [
            ('effect_of_gain', 'T', ['gain', 'loss'], [1, 0]),
            ('effect_of_loss', 'T', ['gain', 'loss'], [0, 1])
            ]
        self.group_level_contrasts = [
            ('group_equal_indifference', 'T', ['equalIndifference', 'equalRange'], [1, 0]),
            ('group_equal_range', 'T', ['equalIndifference', 'equalRange'], [0, 1]),
            ('group_comparison', 'T', ['equalIndifference', 'equalRange'], [-1, 1])
            ]

    def get_preprocessing(self):
        """ No preprocessing has been done by team 4SZ2 """
        return None

    def get_subject_information(event_file):
        """
        Create Bunchs for specifyModel.

        Parameters :
        - event_file : str, file corresponding to the run and the subject to analyze

        Returns :
        - subject_info : list of Bunch for 1st level analysis.
        """
        from nipype.interfaces.base import Bunch

        onsets = []
        durations = []
        amplitudes_gain = []
        amplitudes_loss = []

        with open(event_file, 'rt') as file:
            next(file)  # skip the header

            for line in file:
                info = line.strip().split()
                onsets.append(float(info[0]))
                durations.append(float(info[1]))
                amplitudes_gain.append(float(info[2]))
                amplitudes_loss.append(float(info[3]))

        return [
            Bunch(
                conditions = ['gain', 'loss'],
                onsets = [onsets] * 2,
                durations = [durations] * 2,
                amplitudes = [amplitudes_gain, amplitudes_loss]
                )
            ]

    def get_run_level_analysis(self):
        """
        Create the run level analysis workflow.

        Returns:
            - run_level : nipype.WorkFlow
        """
        # Create run level analysis workflow and connect its nodes
        run_level = Workflow(
            base_dir = self.directories.working_dir,
            name = 'run_level_analysis'
            )

        # IdentityInterface Node - Iterate on subject and runs
        information_source = Node(IdentityInterface(
            fields = ['subject_id', 'run_id']),
            name = 'information_source')
        information_source.iterables = [
            ('subject_id', self.subject_list),
            ('run_id', self.run_list)
            ]

        # SelectFiles - Get necessary files
        templates = {
            'func' : join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'),
            'events' : join('sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_events.tsv')
        }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir
        run_level.connect(information_source, 'subject_id', select_files, 'subject_id')
        run_level.connect(information_source, 'run_id', select_files, 'run_id')

        # IsotropicSmooth Node - Smoothing data
        smoothing_func = Node(IsotropicSmooth(), name = 'smoothing_func')
        smoothing_func.inputs.fwhm = self.fwhm
        run_level.connect(select_files, 'func', smoothing_func, 'in_file')

        # Get Subject Info - get subject specific condition information
        subject_information = Node(Function(
            function = self.get_subject_information,
            input_names = ['event_file'],
            output_names = ['subject_info']
            ), name = 'subject_information')
        run_level.connect(select_files, 'events', subject_information, 'event_file')

        # SpecifyModel Node - Generate run level model
        specify_model = Node(SpecifyModel(), name = 'specify_model')
        specify_model.inputs.high_pass_filter_cutoff = 100
        specify_model.inputs.input_units = 'secs'
        specify_model.inputs.time_repetition = TaskInformation()['RepetitionTime']
        run_level.connect(smoothing_func, 'out_file', specify_model, 'functional_runs')
        run_level.connect(subject_information, 'subject_info', specify_model, 'subject_info')

        # Level1Design Node - Generate files for run level computation
        model_design = Node(Level1Design(), name = 'model_design')
        model_design.inputs.bases = {'dgamma' : {'derivs' : True }}
        model_design.inputs.interscan_interval = TaskInformation()['RepetitionTime']
        model_design.inputs.model_serial_correlations = True
        model_design.inputs.contrasts = self.run_level_contrasts
        run_level.connect(specify_model, 'session_info', model_design, 'session_info')

        # FEATModel Node - Generate run level model
        model_generation = Node(FEATModel(), name = 'model_generation')
        run_level.connect(model_design, 'ev_files', model_generation, 'ev_files')
        run_level.connect(model_design, 'fsf_files', model_generation, 'fsf_file')

        # FILMGLS Node - Estimate first level model
        model_estimate = Node(FILMGLS(), name='model_estimate')
        run_level.connect(smoothing_func, 'out_file', model_estimate, 'in_file')
        run_level.connect(model_generation, 'con_file', model_estimate, 'tcon_file')
        run_level.connect(model_generation, 'design_file', model_estimate, 'design_file')

        # DataSink Node - store the wanted results in the wanted directory
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir
        run_level.connect(model_estimate, 'results_dir', data_sink, 'run_level_analysis.@results')
        run_level.connect(
            model_generation, 'design_file', data_sink, 'run_level_analysis.@design_file')
        run_level.connect(
            model_generation, 'design_image', data_sink, 'run_level_analysis.@design_img')

        # Remove large files, if requested
        if Configuration()['pipelines']['remove_unused_data']:
            remove_smooth = Node(
                InterfaceFactory.create('remove_parent_directory'),
                name = 'remove_smooth')
            run_level.connect(data_sink, 'out_file', remove_smooth, '_')
            run_level.connect(smoothing_func, 'out_file', remove_smooth, 'file_name')

        return run_level

    def get_run_level_outputs(self):
        """ Return the names of the files the run level analysis is supposed to generate. """

        parameters = {
            'run_id' : self.run_list,
            'subject_id' : self.subject_list,
            'contrast_id' : self.contrast_list,
        }
        parameter_sets = product(*parameters.values())
        output_dir = join(self.directories.output_dir,
            'run_level_analysis', '_run_id_{run_id}_subject_id_{subject_id}')
        templates = [
                join(output_dir, 'results', 'cope{contrast_id}.nii.gz'),
                join(output_dir, 'results', 'tstat{contrast_id}.nii.gz'),
                join(output_dir, 'results', 'varcope{contrast_id}.nii.gz'),
                join(output_dir, 'results', 'zstat{contrast_id}.nii.gz')
            ]
        return [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets for template in templates]

    def get_subject_level_analysis(self):
        """ No subject level analysis has been done by team 4SZ2 """
        return None

    @staticmethod
    def get_group_level_regressors(subject_list: list, run_list: list):
        """
        Create dictionary of regressors for two sample t-test group analysis.

        Parameters:
            - subject_list: ids of subject for which to do the analysis
            - run_list: ids of runs for which to do the analysis

        Returns:
            - regressors, dict: containing named lists of regressors.
            - groups, list: group identifiers to distinguish groups in FSL analysis.
        """

        # Create lists containing regressors for each group (equalRange, equalIndifference)
        #  * 1 if the participant is on the group
        #  * 0 otherwise
        equal_range_group = get_group('equalRange')
        equal_indif_group = get_group('equalIndifference')
        equal_range_regressor = [1 if s in equal_range_group else 0 for s in subject_list for _ in run_list]
        equal_indif_regressor = [1 if s in equal_indif_group else 0 for s in subject_list for _ in run_list]

        # Get gender and age of participants
        participants_data = get_participants_information()[['participant_id', 'gender', 'age']]
        participants = participants_data.loc[
            participants_data['participant_id'].isin([f'sub-{s}' for s in subject_list])
            ]
        ages = array(participants['age'])
        genders = array(participants['gender'])

        # Create regressors output
        regressors = dict(
            equalIndifference = equal_indif_regressor,
            equalRange = equal_range_regressor,
            age = [int(a) for a in ages for _ in run_list], 
            gender = [1 if i == 'F' else 0 for i in genders for _ in run_list]
        )

        # Create groups outputs
        groups = [1 if i == 1 else 2 for i in equal_range_regressor]

        return regressors, groups

    def get_group_level_analysis(self):
        """
        Return all workflows for the group level analysis.

        Returns;
            - a list of nipype.WorkFlow
        """
        # Compute the number of participants in the analysis
        nb_subjects = len(self.subject_list)

        # Declare the workflow
        group_level = Workflow(
            base_dir = self.directories.working_dir,
            name = f'group_level_analysis_nsub_{nb_subjects}')

        # Infosource Node - iterate over the contrasts generated by the subject level analysis
        information_source = Node(IdentityInterface(
            fields = ['contrast_id']),
            name = 'information_source')
        information_source.iterables = [('contrast_id', self.contrast_list)]

        # SelectFiles Node - select necessary files
        templates = {
            'cope' : join(self.directories.output_dir,
                'run_level_analysis', '_run_id_*_subject_id_*', 'results',
                'cope{contrast_id}.nii.gz'),
            'varcope' : join(self.directories.output_dir,
                'run_level_analysis', '_run_id_*_subject_id_*', 'results',
                'varcope{contrast_id}.nii.gz'),
            'masks': join('derivatives', 'fmriprep', 'sub-*', 'func',
                'sub-*_task-MGT_run-*_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz')
            }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir
        group_level.connect(information_source, 'contrast_id', select_files, 'contrast_id')

        # Create a function to complete the subject ids out from the get_*_subjects node
        complete_subject_ids = lambda l : [f'_subject_id_{a}' for a in l]
        complete_sub_ids = lambda l : [f'sub-{a}' for a in l]

        # Function Node elements_in_string
        #   Get contrast of parameter estimates (cope) for subjects in a given group
        # Note : using a MapNode with elements_in_string requires using clean_list to remove
        #   None values from the out_list
        get_copes = MapNode(Function(
            function = elements_in_string,
            input_names = ['input_str', 'elements'],
            output_names = ['out_list']
            ),
            name = 'get_copes', iterfield = 'input_str'
        )
        get_copes.inputs.elements = complete_subject_ids(self.subject_list)
        group_level.connect(select_files, 'cope', get_copes, 'input_str')

        # Function Node elements_in_string
        #   Get variance of the estimated copes (varcope) for subjects in a given group
        # Note : using a MapNode with elements_in_string requires using clean_list to remove
        #   None values from the out_list
        get_varcopes = MapNode(Function(
            function = elements_in_string,
            input_names = ['input_str', 'elements'],
            output_names = ['out_list']
            ),
            name = 'get_varcopes', iterfield = 'input_str'
        )
        get_varcopes.inputs.elements = complete_subject_ids(self.subject_list)
        group_level.connect(select_files, 'varcope', get_varcopes, 'input_str')

        # Function Node elements_in_string
        #   Get masks for subjects in a given group
        # Note : using a MapNode with elements_in_string requires using clean_list to remove
        #   None values from the out_list
        get_masks = MapNode(Function(
            function = elements_in_string,
            input_names = ['input_str', 'elements'],
            output_names = ['out_list']
            ),
            name = 'get_masks', iterfield = 'input_str'
        )
        get_masks.inputs.elements = complete_sub_ids(self.subject_list)
        group_level.connect(select_files, 'masks', get_masks, 'input_str')

        # Merge Node - Merge cope files
        merge_copes = Node(Merge(), name = 'merge_copes')
        merge_copes.inputs.dimension = 't'
        group_level.connect(get_copes, ('out_list', clean_list), merge_copes, 'in_files')

        # Merge Masks - Merge mask files
        merge_masks = Node(Merge(), name = 'merge_masks')
        merge_masks.inputs.dimension = 't'
        group_level.connect(get_masks, ('out_list', clean_list), merge_masks, 'in_files')

        # Merge Node - Merge cope files
        merge_varcopes = Node(Merge(), name = 'merge_varcopes')
        merge_varcopes.inputs.dimension = 't'
        group_level.connect(get_varcopes, ('out_list', clean_list), merge_varcopes, 'in_files')

        # MathsCommand Node - Create a global mask by
        #   computing the intersection of all run masks.
        mask_intersection = Node(MathsCommand(), name = 'mask_intersection')
        mask_intersection.inputs.args = '-Tmin -thr 0.9'
        group_level.connect(merge_masks, 'merged_file', mask_intersection, 'in_file')

        # Get regressors for the group level analysis
        regressors, groups = self.get_group_level_regressors(self.subject_list, self.run_list)

        # MultipleRegressDesign Node - Specify model
        specify_model = Node(MultipleRegressDesign(), name = 'specify_model')
        specify_model.inputs.regressors = regressors
        specify_model.inputs.groups = groups
        specify_model.inputs.contrasts = self.group_level_contrasts

        # FLAMEO Node - Estimate model
        estimate_model = Node(FLAMEO(), name = 'estimate_model')
        estimate_model.inputs.run_mode = 'flame1'
        group_level.connect(mask_intersection, 'out_file', estimate_model, 'mask_file')
        group_level.connect(merge_copes, 'merged_file', estimate_model, 'cope_file')
        group_level.connect(merge_varcopes, 'merged_file', estimate_model, 'var_cope_file')
        group_level.connect(specify_model, 'design_mat', estimate_model, 'design_file')
        group_level.connect(specify_model, 'design_con', estimate_model, 't_con_file')
        group_level.connect(specify_model, 'design_grp', estimate_model, 'cov_split_file')

        # Cluster Node - Perform clustering on statistical output
        cluster = MapNode(
            Cluster(),
            name = 'cluster',
            iterfield = ['in_file', 'cope_file'], 
            synchronize = True
            )
        cluster.inputs.threshold = 2.3
        cluster.inputs.out_threshold_file = True
        group_level.connect(estimate_model, 'zstats', cluster, 'in_file')
        group_level.connect(estimate_model, 'copes', cluster, 'cope_file')

        # Datasink Node - Save important files
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir
        group_level.connect(estimate_model, 'zstats', data_sink,
            f'group_level_analysis_nsub_{nb_subjects}.@zstats')
        group_level.connect(estimate_model, 'tstats', data_sink,
            f'group_level_analysis_nsub_{nb_subjects}.@tstats')
        group_level.connect(cluster,'threshold_file', data_sink,
            f'group_level_analysis_nsub_{nb_subjects}.@threshold_file')

        return group_level

    def get_group_level_outputs(self):
        """ Return all names for the files the group level analysis is supposed to generate. """

        parameters = {
            'contrast_id': self.contrast_list,
            'file': [
                '_cluster0/zstat1_threshold.nii.gz',
                '_cluster1/zstat2_threshold.nii.gz',
                'tstat1.nii.gz',
                'tstat2.nii.gz',
                'zstat1.nii.gz',
                'zstat2.nii.gz'
                ]
        }
        parameter_sets = product(*parameters.values())
        template = join(
            self.directories.output_dir,
            'group_level_analysis_nsub_'+f'{len(self.subject_list)}',
            '_contrast_id_{contrast_id}',
            '{file}'
            )
        return [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

    def get_hypotheses_outputs(self):
        """ Return all hypotheses output file names. """

        nb_sub = len(self.subject_list)
        files = [
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_1', '_cluster0', 'zstat1_threshold.nii.gz'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_1', 'zstat1.nii.gz'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_1', '_cluster0', 'zstat1_threshold.nii.gz'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_1', 'zstat1.nii.gz'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_1', '_cluster0', 'zstat1_threshold.nii.gz'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_1', 'zstat1.nii.gz'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_1', '_cluster0', 'zstat1_threshold.nii.gz'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_1', 'zstat1.nii.gz'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_2', '_cluster1', 'zstat2_threshold.nii.gz'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_2', 'zstat2.nii.gz'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_2', '_cluster1', 'zstat2_threshold.nii.gz'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_2', 'zstat2.nii.gz'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_2', '_cluster0', 'zstat1_threshold.nii.gz'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_2', 'zstat1.nii.gz'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_2', '_cluster0', 'zstat1_threshold.nii.gz'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_2', 'zstat1.nii.gz'),
            join(f'group_level_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_2', '_cluster0', 'zstat1_threshold.nii.gz'),
            join(f'group_level_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_2', 'zstat1.nii.gz')
        ]
        return [join(self.directories.output_dir, f) for f in files]
