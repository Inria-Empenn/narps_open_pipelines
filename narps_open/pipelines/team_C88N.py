#!/usr/bin/python
# coding: utf-8

""" Write the work of NARPS' team C88N using Nipype """

from os.path import join
from itertools import product

from nipype import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.spm import (
    Smooth,
    OneSampleTTestDesign, EstimateModel, EstimateContrast,
    Level1Design, TwoSampleTTestDesign, Threshold
    )
from nipype.algorithms.modelgen import SpecifySPMModel
from nipype.algorithms.misc import Gunzip

from narps_open.pipelines import Pipeline
from narps_open.data.task import TaskInformation
from narps_open.data.participants import get_group
from narps_open.core.common import remove_file, list_intersection, elements_in_string, clean_list

class PipelineTeamC88N(Pipeline):
    """ A class that defines the pipeline of team C88N. """

    def __init__(self):
        super().__init__()
        self.fwhm = 8.0
        self.team_id = 'C88N'
        self.model_list = ['gain', 'loss']
        self.subject_level_contrasts_gain = [
            ['effect_of_gain', 'T', ['trialxloss^1', 'trialxgain^1'], [0, 1]]
        ]
        self.subject_level_contrasts_loss = [
            ['positive_effect_of_loss', 'T', ['trialxgain^1', 'trialxloss^1'], [0, 1]],
            ['negative_effect_of_loss', 'T', ['trialxgain^1', 'trialxloss^1'], [0, -1]]
        ]

    def get_preprocessing(self):
        """ No preprocessing has been done by team C88N """
        return None

    def get_run_level_analysis(self):
        """ No run level analysis has been done by team C88N """
        return None

    # @staticmethod # Starting python 3.10, staticmethod should be used here
    # Otherwise it produces a TypeError: 'staticmethod' object is not callable
    def get_subject_information(event_files: list, model: str):
        """ Create Bunchs for SpecifySPMModel.

        Parameters :
        - event_files: list of str, list of events files (one per run) for the subject
        - model: str, either 'gain' or 'loss'

        Returns :
        - subject_information : list of Bunch for 1st level analysis.
        """
        from nipype.interfaces.base import Bunch

        subject_information = []

        # Create on Bunch per run
        for event_file in event_files:

            # Create empty lists
            onsets = []
            durations = []
            weights_gain = []
            weights_loss = []

            # Parse event file
            with open(event_file, 'rt') as file:
                next(file)  # skip the header

                for line in file:
                    info = line.strip().split()
                    if 'NoResp' not in info[5]:
                        onsets.append(float(info[0]))
                        durations.append(0.0)
                        weights_gain.append(float(info[2]))
                        weights_loss.append(float(info[3]))

            # Create Bunch
            if model == 'gain':
                parametric_modulation_bunch = Bunch(
                    name = ['loss', 'gain'],
                    poly = [1, 1],
                    param = [weights_loss, weights_gain]
                    )
            elif model == 'loss':
                parametric_modulation_bunch = Bunch(
                    name = ['gain', 'loss'],
                    poly = [1, 1],
                    param = [weights_gain, weights_loss]
                    )
            else:
                raise AttributeError

            subject_information.append(
                Bunch(
                    conditions = ['trial'],
                    onsets = [onsets],
                    durations = [durations],
                    amplitudes = None,
                    tmod = None,
                    pmod = [parametric_modulation_bunch],
                    regressor_names = None,
                    regressors = None)
                )

        return subject_information

    def get_subject_level_analysis(self):
        """
        Create the subject level analysis workflow.

        Returns:
            - subject_level_analysis : nipype.WorkFlow
        """
        # Infosource Node - To iterate on subjects
        infosource = Node(IdentityInterface(
            fields = ['subject_id']),
            name = 'infosource')
        infosource.iterables = [('subject_id', self.subject_list)]

        # Templates to select files
        template = {
            # Functional MRI
            'func' : join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-*_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'),

            # Event file
            'event' : join('sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-*_events.tsv')
        }

        # SelectFiles - to select necessary files
        select_files = Node(SelectFiles(template), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir

        # DataSink - store the wanted results in the wanted repository
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir

        # Gunzip - gunzip files because SPM do not use .nii.gz files
        gunzip_func = MapNode(Gunzip(),
            name = 'gunzip_func',
            iterfield = ['in_file'])

        # Smoothing - smoothing node
        smoothing = Node(Smooth(), name = 'smoothing')
        smoothing.inputs.fwhm = self.fwhm

        # Function node get_subject_information - get subject specific condition information
        subject_infos_gain = Node(Function(
            function = self.get_subject_information,
            input_names = ['event_files', 'model'],
            output_names = ['subject_info']
            ),
            name = 'subject_infos_gain')
        subject_infos_gain.inputs.model = 'gain'

        subject_infos_loss = Node(Function(
            function = self.get_subject_information,
            input_names = ['event_files', 'model'],
            output_names = ['subject_info']
            ),
            name = 'subject_infos_loss')
        subject_infos_loss.inputs.model = 'loss'

        # SpecifyModel - Generates SPM-specific Model
        specify_model_gain = Node(SpecifySPMModel(), name = 'specify_model_gain')
        specify_model_gain.inputs.concatenate_runs = True
        specify_model_gain.inputs.input_units = 'secs'
        specify_model_gain.inputs.output_units = 'secs'
        specify_model_gain.inputs.time_repetition = TaskInformation()['RepetitionTime']
        specify_model_gain.inputs.high_pass_filter_cutoff = 128

        specify_model_loss = Node(SpecifySPMModel(), name = 'specify_model_loss')
        specify_model_loss.inputs.concatenate_runs = True
        specify_model_loss.inputs.input_units = 'secs'
        specify_model_loss.inputs.output_units = 'secs'
        specify_model_loss.inputs.time_repetition = TaskInformation()['RepetitionTime']
        specify_model_loss.inputs.high_pass_filter_cutoff = 128

        # Level1Design - Generates an SPM design matrix
        model_design_gain = Node(Level1Design(), name = 'model_design_gain')
        model_design_gain.inputs.bases = {'hrf': {'derivs': [0, 0]}}
        model_design_gain.inputs.timing_units = 'secs'
        model_design_gain.inputs.interscan_interval = TaskInformation()['RepetitionTime']

        model_design_loss = Node(Level1Design(), name = 'model_design_loss')
        model_design_loss.inputs.bases = {'hrf': {'derivs': [0, 0]}}
        model_design_loss.inputs.timing_units = 'secs'
        model_design_loss.inputs.interscan_interval = TaskInformation()['RepetitionTime']

        # EstimateModel - estimate the parameters of the model
        model_estimate_gain = Node(EstimateModel(), name = 'model_estimate_gain')
        model_estimate_gain.inputs.estimation_method = {'Classical': 1}

        model_estimate_loss = Node(EstimateModel(), name = 'model_estimate_loss')
        model_estimate_loss.inputs.estimation_method = {'Classical': 1}

        # EstimateContrast - estimates contrasts
        contrast_estimate_gain = Node(EstimateContrast(), name = 'contrast_estimate_gain')
        contrast_estimate_gain.inputs.contrasts = self.subject_level_contrasts_gain

        contrast_estimate_loss = Node(EstimateContrast(), name = 'contrast_estimate_loss')
        contrast_estimate_loss.inputs.contrasts = self.subject_level_contrasts_loss

        # Function node remove_gunzip_files - remove output of the gunzip node
        remove_gunzip_files = MapNode(Function(
            function = remove_file,
            input_names = ['_', 'file_name'],
            output_names = []),
            name = 'remove_gunzip_files', iterfield = 'file_name')

        # Function node remove_smoothed_files - remove output of the smoothing node
        remove_smoothed_files = MapNode(Function(
            function = remove_file,
            input_names = ['_', 'file_name'],
            output_names = []),
            name = 'remove_smoothed_files', iterfield = 'file_name')

        # Create l1 analysis workflow and connect its nodes
        subject_level_analysis = Workflow(
            base_dir = self.directories.working_dir, name = 'subject_level_analysis'
            )
        subject_level_analysis.connect([
            (infosource, select_files, [('subject_id', 'subject_id')]),
            (select_files, subject_infos_gain, [('event','event_files')]),
            (select_files, subject_infos_loss, [('event','event_files')]),
            (subject_infos_gain, specify_model_gain, [('subject_info', 'subject_info')]),
            (subject_infos_loss, specify_model_loss, [('subject_info', 'subject_info')]),
            (select_files, gunzip_func, [('func', 'in_file')]),
            (gunzip_func, smoothing, [('out_file', 'in_files')]),
            (gunzip_func, remove_gunzip_files, [('out_file', 'file_name')]),
            (smoothing, remove_gunzip_files, [('smoothed_files', '_')]),
            (smoothing, specify_model_gain, [('smoothed_files', 'functional_runs')]),
            (smoothing, specify_model_loss, [('smoothed_files', 'functional_runs')]),
            (smoothing, remove_smoothed_files, [('smoothed_files', 'file_name')]),
            (specify_model_gain, model_design_gain, [('session_info', 'session_info')]),
            (specify_model_loss, model_design_loss, [('session_info', 'session_info')]),
            (model_design_gain, model_estimate_gain, [('spm_mat_file', 'spm_mat_file')]),
            (model_design_loss, model_estimate_loss, [('spm_mat_file', 'spm_mat_file')]),
            (model_estimate_gain, contrast_estimate_gain, [
                ('spm_mat_file', 'spm_mat_file'),
                ('beta_images', 'beta_images'),
                ('residual_image', 'residual_image')]),
            (model_estimate_loss, contrast_estimate_loss, [
                ('spm_mat_file', 'spm_mat_file'),
                ('beta_images', 'beta_images'),
                ('residual_image', 'residual_image')]),
            (contrast_estimate_gain, data_sink, [
                ('con_images', 'subject_level_analysis_gain.@con_images'),
                ('spmT_images', 'subject_level_analysis_gain.@spmT_images'),
                ('spm_mat_file', 'subject_level_analysis_gain.@spm_mat_file')]),
            (contrast_estimate_loss, data_sink, [
                ('con_images', 'subject_level_analysis_loss.@con_images'),
                ('spmT_images', 'subject_level_analysis_loss.@spmT_images'),
                ('spm_mat_file', 'subject_level_analysis_loss.@spm_mat_file')]),
            (contrast_estimate_gain, remove_smoothed_files, [('spmT_images', '_')])
            ])

        return subject_level_analysis

    def get_subject_level_outputs(self):
        """ Return the names of the files the subject level analysis is supposed to generate. """

        # Handle gain files
        templates = [join(
            self.directories.output_dir,
            'subject_level_analysis_gain', '_subject_id_{subject_id}', 'con_0001.nii')]
        templates += [join(
            self.directories.output_dir,
            'subject_level_analysis_gain', '_subject_id_{subject_id}', 'SPM.mat')]
        templates += [join(
            self.directories.output_dir,
            'subject_level_analysis_gain', '_subject_id_{subject_id}', 'spmT_0001.nii')]

        # Handle loss files
        contrast_list = ['0001', '0002']
        templates += [join(
            self.directories.output_dir,
            'subject_level_analysis_loss', '_subject_id_{subject_id}', f'con_{contrast_id}.nii')\
            for contrast_id in contrast_list]
        templates += [join(
            self.directories.output_dir,
            'subject_level_analysis_loss', '_subject_id_{subject_id}', 'SPM.mat')]
        templates += [join(
            self.directories.output_dir,
            'subject_level_analysis_loss', '_subject_id_{subject_id}', f'spmT_{contrast_id}.nii')\
            for contrast_id in contrast_list]

        # Format with subject_ids
        return_list = []
        for template in templates:
            return_list += [template.format(subject_id = s) for s in self.subject_list]

        return return_list

    def get_group_level_analysis(self):
        """
        Return all workflows for the group level analysis.

        Returns;
            - a list of nipype.WorkFlow
        """
        return_list = []

        self.model_list = ['gain', 'loss']
        self.contrast_list = ['0001']
        return_list.append(self.get_group_level_analysis_sub_workflow('equalRange'))
        return_list.append(self.get_group_level_analysis_sub_workflow('equalIndifference'))

        self.model_list = ['loss']
        self.contrast_list = ['0001']
        return_list.append(self.get_group_level_analysis_sub_workflow('groupComp'))

        self.model_list = ['loss']
        self.contrast_list = ['0002']
        return_list.append(self.get_group_level_analysis_sub_workflow('equalRange'))
        return_list.append(self.get_group_level_analysis_sub_workflow('equalIndifference'))

        return return_list

    def get_group_level_analysis_sub_workflow(self, method):
        """
        Return a workflow for the group level analysis.

        Parameters:
            - method: one of 'equalRange', 'equalIndifference' or 'groupComp'

        Returns:
            - group_level_analysis: nipype.WorkFlow
        """
        # Compute the number of participants used to do the analysis
        nb_subjects = len(self.subject_list)

        # Infosource - iterate over the list of contrasts
        information_source = Node(IdentityInterface(
            fields = ['model_type', 'contrast_id']),
            name = 'information_source')
        information_source.iterables = [
            ('model_type', self.model_list),
            ('contrast_id', self.contrast_list)
            ]

        # SelectFiles Node
        templates = {
            # Contrast files for all participants
            'contrasts' : join(self.directories.output_dir,
                'subject_level_analysis_{model_type}', '_subject_id_*', 'con_{contrast_id}.nii'
                )
        }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir
        select_files.inputs.force_list = True

        # Datasink - save important files
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir

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

        # Create a function to complete the subject ids out from the get_equal_*_subjects nodes
        #   If not complete, subject id '001' in search patterns
        #   would match all contrast files with 'con_0001.nii'.
        complete_subject_ids = lambda l : [f'_subject_id_{a}' for a in l]

        # Function Node elements_in_string
        #   Get contrast files for required subjects
        # Note : using a MapNode with elements_in_string requires using clean_list to remove
        #   None values from the out_list
        get_contrasts = MapNode(Function(
            function = elements_in_string,
            input_names = ['input_str', 'elements'],
            output_names = ['out_list']
            ),
            name = 'get_contrasts', iterfield = 'input_str'
        )

        # Estimate model
        estimate_model = Node(EstimateModel(), name = 'estimate_model')
        estimate_model.inputs.estimation_method = {'Classical':1}

        # Estimate contrasts
        estimate_contrast = Node(EstimateContrast(), name = 'estimate_contrast')
        estimate_contrast.inputs.group_contrast = True

        # Create thresholded maps
        threshold = MapNode(Threshold(), name = 'threshold',
            iterfield = ['stat_image', 'contrast_index'])
        threshold.inputs.contrast_index = 1
        threshold.inputs.use_topo_fdr = True
        threshold.inputs.use_fwe_correction = False
        threshold.inputs.extent_threshold = 0
        threshold.inputs.height_threshold = 0.001
        threshold.inputs.height_threshold_type = 'p-value'
        threshold.synchronize = True

        group_level_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = f'group_level_analysis_{method}_nsub_{nb_subjects}')
        group_level_analysis.connect([
            (information_source, select_files, [
                ('contrast_id', 'contrast_id'),
                ('model_type', 'model_type')]),
            (select_files, get_contrasts, [('contrasts', 'input_str')]),
            (estimate_model, estimate_contrast, [
                ('spm_mat_file', 'spm_mat_file'),
                ('residual_image', 'residual_image'),
                ('beta_images', 'beta_images')]),
            (estimate_contrast, threshold, [
                ('spm_mat_file', 'spm_mat_file'),
                ('spmT_images', 'stat_image')]),
            (estimate_model, data_sink, [
                ('mask_image', f'group_level_analysis_{method}_nsub_{nb_subjects}.@mask')]),
            (estimate_contrast, data_sink, [
                ('spm_mat_file', f'group_level_analysis_{method}_nsub_{nb_subjects}.@spm_mat'),
                ('spmT_images', f'group_level_analysis_{method}_nsub_{nb_subjects}.@T'),
                ('con_images', f'group_level_analysis_{method}_nsub_{nb_subjects}.@con')]),
            (threshold, data_sink, [
                ('thresholded_map', f'group_level_analysis_{method}_nsub_{nb_subjects}.@thresh')])])

        if method in ('equalRange', 'equalIndifference'):
            estimate_contrast.inputs.contrasts = [
                ('Group', 'T', ['mean'], [1]),
                ('Group', 'T', ['mean'], [-1])
                ]
            threshold.inputs.contrast_index = [1, 2]

            # Specify design matrix
            one_sample_t_test_design = Node(OneSampleTTestDesign(),
                name = 'one_sample_t_test_design')
            group_level_analysis.connect([
                (get_contrasts, one_sample_t_test_design, [
                    (('out_list', clean_list), 'in_files')
                    ]),
                (one_sample_t_test_design, estimate_model, [('spm_mat_file', 'spm_mat_file')])
                ])

        if method == 'equalRange':
            group_level_analysis.connect([
                (get_equal_range_subjects, get_contrasts, [
                    (('out_list', complete_subject_ids), 'elements')
                ])
            ])

        elif method == 'equalIndifference':
            group_level_analysis.connect([
                (get_equal_indifference_subjects, get_contrasts, [
                    (('out_list', complete_subject_ids), 'elements')
                ])
            ])

        elif method == 'groupComp':
            estimate_contrast.inputs.contrasts = [
                ('Eq range vs Eq indiff in loss', 'T', ['Group_{1}', 'Group_{2}'], [1, -1])
                ]
            threshold.inputs.contrast_index = [1]

            # Function Node elements_in_string
            #   Get contrast files for required subjects
            # Note : using a MapNode with elements_in_string requires using clean_list to remove
            #   None values from the out_list
            get_contrasts_2 = MapNode(Function(
                function = elements_in_string,
                input_names = ['input_str', 'elements'],
                output_names = ['out_list']
                ),
                name = 'get_contrasts_2', iterfield = 'input_str'
            )

            # Specify design matrix
            two_sample_t_test_design = Node(TwoSampleTTestDesign(),
                name = 'two_sample_t_test_design')

            group_level_analysis.connect([
                (select_files, get_contrasts_2, [('contrasts', 'input_str')]),
                (get_equal_range_subjects, get_contrasts, [
                    (('out_list', complete_subject_ids), 'elements')
                ]),
                (get_equal_indifference_subjects, get_contrasts_2, [
                    (('out_list', complete_subject_ids), 'elements')
                ]),
                (get_contrasts, two_sample_t_test_design, [
                    (('out_list', clean_list), 'group1_files')
                ]),
                (get_contrasts_2, two_sample_t_test_design, [
                    (('out_list', clean_list), 'group2_files')
                ]),
                (two_sample_t_test_design, estimate_model, [('spm_mat_file', 'spm_mat_file')])
            ])

        return group_level_analysis

    def get_group_level_outputs(self):
        """ Return all names for the files the group level analysis is supposed to generate. """

        # Handle equalRange and equalIndifference

        ## Contrast id 0001
        parameters = {
            'method': ['equalRange', 'equalIndifference'],
            'file': [
                'con_0001.nii', 'con_0002.nii', 'mask.nii', 'SPM.mat',
                'spmT_0001.nii', 'spmT_0002.nii',
                join('_threshold0', 'spmT_0001_thr.nii'), join('_threshold1', 'spmT_0002_thr.nii')
                ],
            'model_type' : ['gain', 'loss'],
            'nb_subjects' : [str(len(self.subject_list))]
        }

        parameter_sets = product(*parameters.values())
        template = join(
            self.directories.output_dir,
            'group_level_analysis_{method}_nsub_{nb_subjects}',
            '_contrast_id_0001_model_type_{model_type}',
            '{file}'
            )

        return_list = [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

        ## Contrast id 0002
        parameters = {
            'method': ['equalRange', 'equalIndifference'],
            'file': [
                'con_0001.nii', 'con_0002.nii', 'mask.nii', 'SPM.mat',
                'spmT_0001.nii', 'spmT_0002.nii',
                join('_threshold0', 'spmT_0001_thr.nii'), join('_threshold1', 'spmT_0002_thr.nii')
                ],
            'nb_subjects' : [str(len(self.subject_list))]
        }

        parameter_sets = product(*parameters.values())
        template = join(
            self.directories.output_dir,
            'group_level_analysis_{method}_nsub_{nb_subjects}',
            '_contrast_id_0002_model_type_loss',
            '{file}'
            )

        return_list += [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

        # Handle groupComp
        parameters = {
            'method': ['groupComp'],
            'file': [
                'con_0001.nii', 'mask.nii', 'SPM.mat', 'spmT_0001.nii',
                join('_threshold0', 'spmT_0001_thr.nii')
                ],
            'nb_subjects' : [str(len(self.subject_list))]
        }
        parameter_sets = product(*parameters.values())
        template = join(
            self.directories.output_dir,
            'group_level_analysis_{method}_nsub_{nb_subjects}',
            '_contrast_id_0001_model_type_loss',
            '{file}'
            )

        return_list += [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

        return return_list

    def get_hypotheses_outputs(self):
        """ Return all hypotheses output file names. """
        nb_sub = len(self.subject_list)
        files = [
            # Hypothesis 1
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_gain', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_gain', 'spmT_0001.nii'),
            # Hypothesis 2
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_gain', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_gain', 'spmT_0001.nii'),
            # Hypothesis 3
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_gain', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_gain', 'spmT_0001.nii'),
            # Hypothesis 4
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_gain', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_gain', 'spmT_0001.nii'),
            # Hypothesis 5
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_loss', '_threshold1', 'spmT_0002_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_loss', 'spmT_0002.nii'),
            # Hypothesis 6
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_loss', '_threshold1', 'spmT_0002_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_loss', 'spmT_0002.nii'),
            # Hypothesis 7
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_loss', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_loss', 'spmT_0001.nii'),
            # Hypothesis 8
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_loss', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_loss', 'spmT_0001.nii'),
            # Hypothesis 9
            join(f'group_level_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_loss', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_loss', 'spmT_0001.nii')
        ]
        return [join(self.directories.output_dir, f) for f in files]
