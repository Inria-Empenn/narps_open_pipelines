#!/usr/bin/python
# coding: utf-8

""" Write the work of NARPS team T54A using Nipype """

from os.path import join
from itertools import product

from nipype import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Function, Split
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.fsl import (
    BET, IsotropicSmooth, Level1Design, FEATModel, L2Model, Merge, FLAMEO,
    FILMGLS, Randomise, MultipleRegressDesign, FSLCommand
    )
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces.fsl.maths import MultiImageMaths

from narps_open.utils.configuration import Configuration
from narps_open.pipelines import Pipeline
from narps_open.data.task import TaskInformation
from narps_open.data.participants import get_group
from narps_open.core.common import list_intersection, elements_in_string, clean_list
from narps_open.core.interfaces import InterfaceFactory

# Setup FSL
FSLCommand.set_default_output_type('NIFTI_GZ')

class PipelineTeamT54A(Pipeline):
    """ A class that defines the pipeline of team T54A """

    def __init__(self):
        super().__init__()
        self.fwhm = 4.0
        self.team_id = 'T54A'
        self.contrast_list = ['1', '2']
        self.run_level_contrasts = [
            ('gain', 'T', ['trial', 'gain', 'loss'], [0, 1, 0]),
            ('loss', 'T', ['trial', 'gain', 'loss'], [0, 0, 1])
            ]

    def get_preprocessing(self):
        """ No preprocessing has been done by team T54A """
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

        condition_names = ['trial', 'gain', 'loss', 'difficulty', 'response', 'missed']
        onsets = {}
        durations = {}
        amplitudes = {}

        for condition in condition_names:
            # Create dictionary items with empty lists
            onsets.update({condition : []})
            durations.update({condition : []})
            amplitudes.update({condition : []})

        with open(event_file, 'rt') as file:
            next(file)  # skip the header

            for line in file:
                info = line.strip().split()

                if info[5] != 'NoResp':
                    onsets['trial'].append(float(info[0]))
                    durations['trial'].append(float(info[4]))
                    amplitudes['trial'].append(1.0)
                    onsets['gain'].append(float(info[0]))
                    durations['gain'].append(float(info[4]))
                    amplitudes['gain'].append(float(info[2]))
                    onsets['loss'].append(float(info[0]))
                    durations['loss'].append(float(info[4]))
                    amplitudes['loss'].append(float(info[3]))
                    onsets['difficulty'].append(float(info[0]))
                    durations['difficulty'].append(float(info[4]))
                    amplitudes['difficulty'].append(
                        abs(0.5 * float(info[2]) - float(info[3]))
                        )
                    onsets['response'].append(float(info[0]) + float(info[4]))
                    durations['response'].append(0.0)
                    amplitudes['response'].append(1.0)
                else:
                    onsets['missed'].append(float(info[0]))
                    durations['missed'].append(0.0)
                    amplitudes['missed'].append(1.0)

        # Check if there where missed trials for this run
        if not onsets['missed']:
            condition_names.remove('missed')

        return [
            Bunch(
                conditions = condition_names,
                onsets = [onsets[k] for k in condition_names],
                durations = [durations[k] for k in condition_names],
                amplitudes = [amplitudes[k] for k in condition_names],
                regressor_names = None,
                regressors = None)
            ]

    def get_parameters_file(filepath, subject_id, run_id, working_dir):
        """
        Create a tsv file with only desired parameters per subject per run.

        Parameters :
        - filepath : path to the subject parameters file (i.e. one per run)
        - subject_id : subject for whom the 1st level analysis is made
        - run_id: run for which the 1st level analysis is made
        - working_dir: str, name of the directory for intermediate results

        Return :
        - parameters_file : paths to new files containing only desired parameters.
        """
        from os import makedirs
        from os.path import join

        from pandas import read_csv, DataFrame
        from numpy import array, transpose

        data_frame = read_csv(filepath, sep = '\t', header=0)
        if 'NonSteadyStateOutlier00' in data_frame.columns:
            temp_list = array([
                data_frame['X'], data_frame['Y'], data_frame['Z'],
                data_frame['RotX'], data_frame['RotY'], data_frame['RotZ'],
                data_frame['NonSteadyStateOutlier00']])
        else:
            temp_list = array([
                data_frame['X'], data_frame['Y'], data_frame['Z'],
                data_frame['RotX'], data_frame['RotY'], data_frame['RotZ']])
        retained_parameters = DataFrame(transpose(temp_list))

        parameters_file = join(working_dir, 'parameters_file',
            f'parameters_file_sub-{subject_id}_run-{run_id}.tsv')

        makedirs(join(working_dir, 'parameters_file'), exist_ok = True)

        with open(parameters_file, 'w') as writer:
            writer.write(retained_parameters.to_csv(
                sep = '\t', index = False, header = False, na_rep = '0.0'))

        return parameters_file

    def get_run_level_analysis(self):
        """
        Create the run level analysis workflow.

        Returns:
            - run_level_analysis : nipype.WorkFlow
        """
        # IdentityInterface Node - To iterate on subject and runs
        information_source = Node(IdentityInterface(
            fields = ['subject_id', 'run_id']),
            name = 'information_source')
        information_source.iterables = [
            ('subject_id', self.subject_list),
            ('run_id', self.run_list)
            ]

        # SelectFiles - to select necessary files
        template = {
            # Parameter file
            'param' : join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_confounds.tsv'),
            # Functional MRI
            'func' : join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
            'sub-{subject_id}_task-MGT_run-{run_id}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'
            ),
            # Event file
            'event' : join('sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_events.tsv')
        }
        select_files = Node(SelectFiles(template), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir

        # DataSink Node - store the wanted results in the wanted directory
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir

        # BET Node - Skullstripping data
        skull_stripping_func = Node(BET(), name = 'skull_stripping_func')
        skull_stripping_func.inputs.frac = 0.3
        skull_stripping_func.inputs.functional = True
        skull_stripping_func.inputs.mask = True

        # IsotropicSmooth Node - Smoothing data
        smoothing_func = Node(IsotropicSmooth(), name = 'smoothing_func')
        smoothing_func.inputs.fwhm = self.fwhm # TODO : Previously set to 6 mm ?

        # Function Node get_subject_infos - Get subject specific condition information
        subject_information = Node(Function(
            function = self.get_subject_information,
            input_names = ['event_file'],
            output_names = ['subject_info']
            ), name = 'subject_information')

        # SpecifyModel Node - Generate run level model
        specify_model = Node(SpecifyModel(), name = 'specify_model')
        specify_model.inputs.high_pass_filter_cutoff = 100
        specify_model.inputs.input_units = 'secs'
        specify_model.inputs.time_repetition = TaskInformation()['RepetitionTime']

        # Function Node get_parameters_file - Get files with movement parameters
        parameters = Node(Function(
            function = self.get_parameters_file,
            input_names = ['filepath', 'subject_id', 'run_id', 'working_dir'],
            output_names = ['parameters_file']),
            name = 'parameters')
        parameters.inputs.working_dir = self.directories.working_dir

        # Level1Design Node - Generate files for run level computation
        model_design = Node(Level1Design(), name = 'model_design')
        model_design.inputs.bases = {'dgamma':{'derivs' : True}}
        model_design.inputs.interscan_interval = TaskInformation()['RepetitionTime']
        model_design.inputs.model_serial_correlations = True
        model_design.inputs.contrasts = self.run_level_contrasts

        # FEATModel Node - Generate run level model
        model_generation = Node(FEATModel(), name = 'model_generation')

        # FILMGLS Node - Estimate first level model
        model_estimate = Node(FILMGLS(), name = 'model_estimate')

        # Create l1 analysis workflow and connect its nodes
        run_level_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = 'run_level_analysis'
            )
        run_level_analysis.connect([
            (information_source, select_files, [
                ('subject_id', 'subject_id'),
                ('run_id', 'run_id')]),
            (select_files, subject_information, [('event', 'event_file')]),
            (select_files, parameters, [('param', 'filepath')]),
            (information_source, parameters, [
                ('subject_id', 'subject_id'),
                ('run_id', 'run_id')]),
            (select_files, skull_stripping_func, [('func', 'in_file')]),
            (skull_stripping_func, smoothing_func, [('out_file', 'in_file')]),
            (parameters, specify_model, [('parameters_file', 'realignment_parameters')]),
            (smoothing_func, specify_model, [('out_file', 'functional_runs')]),
            (subject_information, specify_model, [('subject_info', 'subject_info')]),
            (specify_model, model_design, [('session_info', 'session_info')]),
            (model_design, model_generation, [
                ('ev_files', 'ev_files'),
                ('fsf_files', 'fsf_file')]),
            (smoothing_func, model_estimate, [('out_file', 'in_file')]),
            (model_generation, model_estimate, [
                ('con_file', 'tcon_file'),
                ('design_file', 'design_file')]),
            (model_estimate, data_sink, [('results_dir', 'run_level_analysis.@results')]),
            (model_generation, data_sink, [
                ('design_file', 'run_level_analysis.@design_file'),
                ('design_image', 'run_level_analysis.@design_img')]),
            (skull_stripping_func, data_sink, [('mask_file', 'run_level_analysis.@skullstriped')])
        ])

        # Remove large files, if requested
        if Configuration()['pipelines']['remove_unused_data']:

            # Remove Node - Remove skullstriped func files once they are no longer needed
            remove_skullstrip = Node(
                InterfaceFactory.create('remove_parent_directory'),
                name = 'remove_skullstrip')

            # Remove Node - Remove smoothed files once they are no longer needed
            remove_smooth = Node(
                InterfaceFactory.create('remove_parent_directory'),
                name = 'remove_smooth')

            # Add connections
            run_level_analysis.connect([
                (data_sink, remove_skullstrip, [('out_file', '_')]),
                (skull_stripping_func, remove_skullstrip, [('out_file', 'file_name')]),
                (model_estimate, remove_smooth, [('results_dir', '_')]),
                (smoothing_func, remove_smooth, [('out_file', 'file_name')])
                ])

        return run_level_analysis

    def get_run_level_outputs(self):
        """ Return the names of the files the run level analysis is supposed to generate. """

        return_list = []
        output_dir = join(self.directories.output_dir, 'run_level_analysis',
            '_run_id_{run_id}_subject_id_{subject_id}')

        # Handle results dir
        parameters = {
            'run_id' : self.run_list,
            'subject_id' : self.subject_list,
            'contrast_id' : self.contrast_list
        }
        parameter_sets = product(*parameters.values())
        templates = [
            join(output_dir, 'results', 'cope{contrast_id}.nii.gz'),
            join(output_dir, 'results', 'tstat{contrast_id}.nii.gz'),
            join(output_dir, 'results', 'varcope{contrast_id}.nii.gz'),
            join(output_dir, 'results', 'zstat{contrast_id}.nii.gz')
            ]
        return_list = [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets for template in templates]

        # Handle mask file
        parameters = {
            'run_id' : self.run_list,
            'subject_id' : self.subject_list,
        }
        parameter_sets = product(*parameters.values())
        template = join(output_dir, 'sub-{subject_id}_task-MGT_run-{run_id}_bold_space-MNI152NLin2009cAsym_preproc_brain_mask.nii.gz')
        return_list += [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

        return return_list

    def get_subject_level_analysis(self):
        """
        Create the subject level analysis workflow.

        Returns:
        - subject_level_analysis : nipype.WorkFlow
        """
        # Infosource Node - To iterate on subject and runs
        information_source = Node(IdentityInterface(
            fields = ['subject_id', 'contrast_id']),
            name = 'information_source')
        information_source.iterables = [
            ('subject_id', self.subject_list),
            ('contrast_id', self.contrast_list)
            ]

        # Templates to select files node
        templates = {
            'cope' : join(self.directories.output_dir,
                'run_level_analysis', '_run_id_*_subject_id_{subject_id}', 'results',
                'cope{contrast_id}.nii.gz'),
            'varcope' : join(self.directories.output_dir,
                'run_level_analysis', '_run_id_*_subject_id_{subject_id}', 'results',
                'varcope{contrast_id}.nii.gz'),
            'masks': join(self.directories.output_dir,
                'run_level_analysis', '_run_id_*_subject_id_{subject_id}',
                'sub-{subject_id}_task-MGT_run-*_bold_space-MNI152NLin2009cAsym_preproc_brain_mask.nii.gz')
        }

        # SelectFiles Node - to select necessary files
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.results_dir

        # DataSink Node - store the wanted results in the wanted directory
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir

        # L2Model Node - Generate subject specific second level model
        generate_model = Node(L2Model(), name = 'generate_model')
        generate_model.inputs.num_copes = len(self.run_list)

        # Merge Node - Merge copes files for each subject
        merge_copes = Node(Merge(), name = 'merge_copes')
        merge_copes.inputs.dimension = 't'

        # Merge Node - Merge varcopes files for each subject
        merge_varcopes = Node(Merge(), name = 'merge_varcopes')
        merge_varcopes.inputs.dimension = 't'

        # Split Node - Split mask list to serve them as inputs of the MultiImageMaths node.
        split_masks = Node(Split(), name = 'split_masks')
        split_masks.inputs.splits = [1, len(self.run_list) - 1]
        split_masks.inputs.squeeze = True # Unfold one-element splits removing the list

        # MultiImageMaths Node - Create a subject mask by
        #   computing the intersection of all run masks.
        mask_intersection = Node(MultiImageMaths(), name = 'mask_intersection')
        mask_intersection.inputs.op_string = '-mul %s ' * (len(self.run_list) - 1)

        # FLAMEO Node - Estimate model
        estimate_model = Node(FLAMEO(), name = 'estimate_model')
        estimate_model.inputs.run_mode = 'flame1'

        # Second level (single-subject, mean of all four scans) analyses: Fixed effects analysis.
        subject_level_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = 'subject_level_analysis')
        subject_level_analysis.connect([
            (information_source, select_files, [
                ('subject_id', 'subject_id'),
                ('contrast_id', 'contrast_id')]),
            (select_files, merge_copes, [('cope', 'in_files')]),
            (select_files, merge_varcopes, [('varcope', 'in_files')]),
            (select_files, split_masks, [('masks', 'inlist')]),
            (split_masks, mask_intersection, [('out1', 'in_file')]),
            (split_masks, mask_intersection, [('out2', 'operand_files')]),
            (mask_intersection, estimate_model, [('out_file', 'mask_file')]),
            (merge_copes, estimate_model, [('merged_file', 'cope_file')]),
            (merge_varcopes, estimate_model, [('merged_file', 'var_cope_file')]),
            (generate_model, estimate_model, [
                ('design_mat', 'design_file'),
                ('design_con', 't_con_file'),
                ('design_grp', 'cov_split_file')]),
            (mask_intersection, data_sink, [('out_file', 'subject_level_analysis.@mask')]),
            (estimate_model, data_sink, [
                ('zstats', 'subject_level_analysis.@stats'),
                ('tstats', 'subject_level_analysis.@tstats'),
                ('copes', 'subject_level_analysis.@copes'),
                ('var_copes', 'subject_level_analysis.@varcopes')])])

        return subject_level_analysis

    def get_subject_level_outputs(self):
        """ Return the names of the files the subject level analysis is supposed to generate. """

        parameters = {
            'contrast_id' : self.contrast_list,
            'subject_id' : self.subject_list,
        }
        parameter_sets = product(*parameters.values())
        output_dir = join(self.directories.output_dir, 'subject_level_analysis',
            '_contrast_id_{contrast_id}_subject_id_{subject_id}')            
        templates = [
            join(output_dir, 'cope1.nii.gz'),
            join(output_dir, 'tstat1.nii.gz'),
            join(output_dir, 'varcope1.nii.gz'),
            join(output_dir, 'zstat1.nii.gz'),
            join(output_dir, 'sub-{subject_id}_task-MGT_run-01_bold_space-MNI152NLin2009cAsym_preproc_brain_mask_maths.nii.gz')
            ]

        return [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets for template in templates]

    def get_one_sample_t_test_regressors(subject_list: list) -> dict:
        """
        Create dictionary of regressors for one sample t-test group analysis.

        Parameters:
            - subject_list: ids of subject in the group for which to do the analysis

        Returns:
            - dict containing named lists of regressors.
        """

        return dict(group_mean = [1 for _ in subject_list])

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
        # Infosource Node - iterate over the contrasts generated by the subject level analysis
        information_source = Node(IdentityInterface(
            fields = ['contrast_id']),
            name = 'information_source')
        information_source.iterables = [('contrast_id', self.contrast_list)]

        # SelectFiles Node - select necessary files
        templates = {
            'cope' : join(self.directories.output_dir,
                'subject_level_analysis', '_contrast_id_{contrast_id}_subject_id_*',
                'cope1.nii.gz'),
            'varcope' : join(self.directories.output_dir,
                'subject_level_analysis', '_contrast_id_{contrast_id}_subject_id_*',
                'varcope1.nii.gz'),
            'masks': join(self.directories.output_dir,
                'subject_level_analysis', '_contrast_id_1_subject_id_*',
                'sub-*_task-MGT_run-*_bold_space-MNI152NLin2009cAsym_preproc_brain_mask_maths.nii.gz')
            }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.results_dir

        # Datasink Node - save important files
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir

        # Function Node elements_in_string
        #   Get contrast of parameter estimates (cope) for these subjects
        # Note : using a MapNode with elements_in_string requires using clean_list to remove
        #   None values from the out_list
        get_copes = MapNode(Function(
            function = elements_in_string,
            input_names = ['input_str', 'elements'],
            output_names = ['out_list']
            ),
            name = 'get_copes', iterfield = 'input_str'
        )

        # Function Node elements_in_string
        #   Get variance of the estimated copes (varcope) for these subjects
        # Note : using a MapNode with elements_in_string requires using clean_list to remove
        #   None values from the out_list
        get_varcopes = MapNode(Function(
            function = elements_in_string,
            input_names = ['input_str', 'elements'],
            output_names = ['out_list']
            ),
            name = 'get_varcopes', iterfield = 'input_str'
        )

        # Merge Node - Merge cope files
        merge_copes = Node(Merge(), name = 'merge_copes')
        merge_copes.inputs.dimension = 't'

        # Merge Node - Merge cope files
        merge_varcopes = Node(Merge(), name = 'merge_varcopes')
        merge_varcopes.inputs.dimension = 't'

        # Split Node - Split mask list to serve them as inputs of the MultiImageMaths node.
        split_masks = Node(Split(), name = 'split_masks')
        split_masks.inputs.splits = [1, len(self.subject_list) - 1]
        split_masks.inputs.squeeze = True # Unfold one-element splits removing the list

        # MultiImageMaths Node - Create a subject mask by
        #   computing the intersection of all run masks.
        mask_intersection = Node(MultiImageMaths(), name = 'mask_intersection')
        mask_intersection.inputs.op_string = '-mul %s ' * (len(self.subject_list) - 1)

        # MultipleRegressDesign Node - Specify model
        specify_model = Node(MultipleRegressDesign(), name = 'specify_model')

        # FLAMEO Node - Estimate model
        estimate_model = Node(FLAMEO(), name = 'estimate_model')
        estimate_model.inputs.run_mode = 'flame1'

        # Randomise Node -
        randomise = Node(Randomise(), name = 'randomise')
        randomise.inputs.num_perm = 10000
        randomise.inputs.tfce = True
        randomise.inputs.vox_p_values = True
        randomise.inputs.c_thresh = 0.05
        randomise.inputs.tfce_E = 0.01

        # Compute the number of participants used to do the analysis
        nb_subjects = len(self.subject_list)

        # Declare the workflow
        group_level_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = f'group_level_analysis_{method}_nsub_{nb_subjects}')
        group_level_analysis.connect([
            (information_source, select_files, [('contrast_id', 'contrast_id')]),
            (select_files, get_copes, [('cope', 'input_str')]),
            (select_files, get_varcopes, [('varcope', 'input_str')]),
            (get_copes, merge_copes, [(('out_list', clean_list), 'in_files')]),
            (get_varcopes, merge_varcopes,[(('out_list', clean_list), 'in_files')]),
            (select_files, split_masks, [('masks', 'inlist')]),
            (split_masks, mask_intersection, [('out1', 'in_file')]),
            (split_masks, mask_intersection, [('out2', 'operand_files')]),
            (mask_intersection, estimate_model, [('out_file', 'mask_file')]),
            (mask_intersection, randomise, [('out_file', 'mask')]),
            (merge_copes, estimate_model, [('merged_file', 'cope_file')]),
            (merge_varcopes, estimate_model, [('merged_file', 'var_cope_file')]),
            (specify_model, estimate_model, [
                ('design_mat', 'design_file'),
                ('design_con', 't_con_file'),
                ('design_grp', 'cov_split_file')
                ]),
            (merge_copes, randomise, [('merged_file', 'in_file')]),
            (specify_model, randomise, [
                ('design_mat', 'design_mat'),
                ('design_con', 'tcon')
                ]),
            (randomise, data_sink, [
                ('t_corrected_p_files', f'group_level_analysis_{method}_nsub_{nb_subjects}.@tcorpfile'),
                ('tstat_files', f'group_level_analysis_{method}_nsub_{nb_subjects}.@tstat')
                ]),
            (estimate_model, data_sink, [
                ('zstats', f'group_level_analysis_{method}_nsub_{nb_subjects}.@zstats'),
                ('tstats', f'group_level_analysis_{method}_nsub_{nb_subjects}.@tstats')
                ])
            ])

        if method in ('equalIndifference', 'equalRange'):

            # Setup a one sample t-test
            specify_model.inputs.contrasts = [
                ['group_mean', 'T', ['group_mean'], [1]],
                ['group_mean_neg', 'T', ['group_mean'], [-1]]
                ]

            # Function Node get_group_subjects - Get subjects in the group and in the subject_list
            get_group_subjects = Node(Function(
                function = list_intersection,
                input_names = ['list_1', 'list_2'],
                output_names = ['out_list']
                ),
                name = 'get_group_subjects'
            )
            get_group_subjects.inputs.list_1 = get_group(method)
            get_group_subjects.inputs.list_2 = self.subject_list

            # Function Node get_one_sample_t_test_regressors
            #   Get regressors in the equalRange and equalIndifference method case
            regressors_one_sample = Node(
                Function(
                    function = self.get_one_sample_t_test_regressors,
                    input_names = ['subject_list'],
                    output_names = ['regressors']
                ),
                name = 'regressors_one_sample',
            )

            # Add missing connections
            group_level_analysis.connect([
                (get_group_subjects, get_copes, [('out_list', 'elements')]),
                (get_group_subjects, get_varcopes, [('out_list', 'elements')]),
                (get_group_subjects, regressors_one_sample, [('out_list', 'subject_list')]),
                (regressors_one_sample, specify_model, [('regressors', 'regressors')])
            ])

        elif method == 'groupComp':

            # Select copes and varcopes corresponding to the selected subjects
            #   Indeed the SelectFiles node asks for all (*) subjects available
            get_copes.inputs.elements = self.subject_list
            get_varcopes.inputs.elements = self.subject_list

            # Setup a two sample t-test
            specify_model.inputs.contrasts = [
                ['equalRange_sup', 'T', ['equalRange', 'equalIndifference'], [1, -1]]
            ]

            # Function Node get_equal_range_subjects
            #   Get subjects in the equalRange group and in the subject_list
            get_equal_range_subjects = Node(Function(
                function = list_intersection,
                input_names = ['list_1', 'list_2'],
                output_names = ['out_list']
                ),
                name = 'get_equal_range_subjects'
            )
            get_equal_range_subjects.inputs.list_1 = get_group('equalRange')
            get_equal_range_subjects.inputs.list_2 = self.subject_list

            # Function Node get_equal_indifference_subjects
            #   Get subjects in the equalIndifference group and in the subject_list
            get_equal_indifference_subjects = Node(Function(
                function = list_intersection,
                input_names = ['list_1', 'list_2'],
                output_names = ['out_list']
                ),
                name = 'get_equal_indifference_subjects'
            )
            get_equal_indifference_subjects.inputs.list_1 = get_group('equalIndifference')
            get_equal_indifference_subjects.inputs.list_2 = self.subject_list

            # Function Node get_two_sample_t_test_regressors
            #   Get regressors in the groupComp method case
            regressors_two_sample = Node(
                Function(
                    function = self.get_two_sample_t_test_regressors,
                    input_names = [
                        'equal_range_ids',
                        'equal_indifference_ids',
                        'subject_list',
                    ],
                    output_names = ['regressors', 'groups']
                ),
                name = 'regressors_two_sample',
            )
            regressors_two_sample.inputs.subject_list = self.subject_list

            # Add missing connections
            group_level_analysis.connect([
                (get_equal_range_subjects, regressors_two_sample, [
                    ('out_list', 'equal_range_ids')
                    ]),
                (get_equal_indifference_subjects, regressors_two_sample, [
                    ('out_list', 'equal_indifference_ids')
                    ]),
                (regressors_two_sample, specify_model, [
                    ('regressors', 'regressors'),
                    ('groups', 'groups')])
            ])

        return group_level_analysis

    def get_group_level_outputs(self):
        """ Return all names for the files the group level analysis is supposed to generate. """

        # Handle equalRange and equalIndifference
        parameters = {
            'contrast_id': self.contrast_list,
            'method': ['equalRange', 'equalIndifference'],
            'file': [
                'randomise_tfce_corrp_tstat1.nii.gz',
                'randomise_tfce_corrp_tstat2.nii.gz',
                'randomise_tstat1.nii.gz',
                'randomise_tstat2.nii.gz',
                'tstat1.nii.gz',
                'tstat2.nii.gz',
                'zstat1.nii.gz',
                'zstat2.nii.gz'
                ],
            'nb_subjects' : [str(len(self.subject_list))]
        }
        parameter_sets = product(*parameters.values())
        template = join(
            self.directories.output_dir,
            'group_level_analysis_{method}_nsub_{nb_subjects}',
            '_contrast_id_{contrast_id}',
            '{file}'
            )

        return_list = [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

        # Handle groupComp
        files = [
            'randomise_tfce_corrp_tstat1.nii.gz',
            'randomise_tstat1.nii.gz',
            'zstat1.nii.gz',
            'tstat1.nii.gz'
            ]

        return_list += [join(
            self.directories.output_dir,
            f'group_level_analysis_groupComp_nsub_{len(self.subject_list)}',
            '_contrast_id_2', f'{file}') for file in files]

        return return_list

    def get_hypotheses_outputs(self):
        """ Return all hypotheses output file names. """

        nb_sub = len(self.subject_list)
        files = [
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_1', 'randomise_tfce_corrp_tstat1.nii.gz'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_1', 'zstat1.nii.gz'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_1', 'randomise_tfce_corrp_tstat1.nii.gz'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_1', 'zstat1.nii.gz'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_1', 'randomise_tfce_corrp_tstat1.nii.gz'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_1', 'zstat1.nii.gz'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_1', 'randomise_tfce_corrp_tstat1.nii.gz'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_1', 'zstat1.nii.gz'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_2', 'randomise_tfce_corrp_tstat2.nii.gz'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_2', 'zstat2.nii.gz'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_2', 'randomise_tfce_corrp_tstat2.nii.gz'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_2', 'zstat2.nii.gz'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_2', 'randomise_tfce_corrp_tstat1.nii.gz'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_2', 'zstat1.nii.gz'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_2', 'randomise_tfce_corrp_tstat1.nii.gz'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_2', 'zstat1.nii.gz'),
            join(f'group_level_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_2', 'randomise_tfce_corrp_tstat1.nii.gz'),
            join(f'group_level_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_2', 'zstat1.nii.gz')
        ]
        return [join(self.directories.output_dir, f) for f in files]
