#!/usr/bin/python
# coding: utf-8

""" Write the work of NARPS' team J7F9 using Nipype """

from os.path import join
from itertools import product

from nipype import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.spm import (
    Smooth, Level1Design, OneSampleTTestDesign, TwoSampleTTestDesign,
    EstimateModel, EstimateContrast, Threshold
    )
from nipype.algorithms.modelgen import SpecifySPMModel
from nipype.algorithms.misc import Gunzip

from narps_open.pipelines import Pipeline
from narps_open.data.task import TaskInformation
from narps_open.data.participants import get_group
from narps_open.core.common import (
    remove_file, list_intersection, elements_in_string, clean_list
    )

class PipelineTeamJ7F9(Pipeline):
    """ A class that defines the pipeline of team J7F9. """

    def __init__(self):
        super().__init__()
        self.fwhm = 8.0
        self.team_id = 'J7F9'
        self.contrast_list = ['0001', '0002', '0003']
        self.subject_level_contrasts = [
            ['trial', 'T', ['trial', 'trialxgain^1', 'trialxloss^1'], [1, 0, 0]],
            ['effect_of_gain', 'T', ['trial', 'trialxgain^1', 'trialxloss^1'], [0, 1, 0]],
            ['effect_of_loss', 'T', ['trial', 'trialxgain^1', 'trialxloss^1'], [0, 0, 1]]
            ]

    def get_preprocessing(self):
        """ No preprocessing has been done by team J7F9 """
        return None

    def get_run_level_analysis(self):
        """ No run level analysis has been done by team J7F9 """
        return None

    def get_subject_information(event_files):
        """
        Create Bunchs for specifySPMModel.

        Parameters :
        - event_files: list of str, list of events files (one per run) for the subject

        Returns :
        - subject_information : list of Bunch for subject level analysis.
        """
        from numpy import mean, ravel
        from nipype.interfaces.base import Bunch

        subject_information = []

        # Create empty dicts
        onsets = {}
        durations = {}
        weights_gain = {}
        weights_loss = {}
        onsets_missed = {}
        durations_missed = {}

        # Run list
        run_list = [str(r).zfill(2) for r in range(1, len(event_files) + 1)]

        # Parse event file
        for run_id, event_file in zip(run_list, event_files):

            # Init empty lists inside directiries
            onsets[run_id] = []
            durations[run_id] = []
            weights_gain[run_id] = []
            weights_loss[run_id] = []
            onsets_missed[run_id] = []
            durations_missed[run_id] = []

            with open(event_file, 'rt') as file:
                next(file)  # skip the header

                for line in file:
                    info = line.strip().split()

                    # Trials
                    onsets[run_id].append(float(info[0]))
                    durations[run_id].append(0.0)
                    weights_gain[run_id].append(float(info[2]))
                    weights_loss[run_id].append(float(info[3]))

                    # Missed trials
                    if float(info[4]) < 0.1 or 'NoResp' in info[5]:
                        onsets_missed[run_id].append(float(info[0]))
                        durations_missed[run_id].append(0.0)

        # Compute mean weight values across all runs
        all_weights_gain = []
        for value in weights_gain.values():
            all_weights_gain += value
        mean_gain_weight = mean(all_weights_gain)

        all_weights_loss = []
        for value in weights_loss.values():
            all_weights_loss += value
        mean_loss_weight = mean(all_weights_loss)

        # Check if there are any missed trials across all runs
        missed_trials = any(t for t in onsets_missed.values())

        # Create one Bunch per run
        for run_id in run_list:

            # Mean center gain and loss weights
            for element_id, element in enumerate(weights_gain[run_id]):
                weights_gain[run_id][element_id] = element - mean_gain_weight
            for element_id, element in enumerate(weights_loss[run_id]):
                weights_loss[run_id][element_id] = element - mean_loss_weight

            # Fill Bunch
            subject_information.append(
                Bunch(
                    conditions = ['trial', 'missed'] if missed_trials else ['trial'],
                    onsets = [onsets[run_id], onsets_missed[run_id]] if missed_trials\
                        else [onsets[run_id]],
                    durations = [durations[run_id], durations_missed[run_id]] if missed_trials\
                        else [durations[run_id]],
                    amplitudes = None,
                    tmod = None,
                    pmod = [
                        Bunch(
                            name = ['gain', 'loss'],
                            poly = [1, 1],
                            param = [weights_gain[run_id], weights_loss[run_id]]
                        )
                    ],
                    regressor_names = None,
                    regressors = None
                )
            )

        return subject_information

    def get_confounds_file(filepath, subject_id, run_id, working_dir):
        """
        Create a new tsv files with only desired confounds per subject per run.

        Parameters :
        - filepath : path to the subject confounds file
        - subject_id : related subject id
        - run_id : related run id
    	- working_dir: str, name of the directory for intermediate results

        Return :
        - confounds_file : path to new file containing only desired confounds
        """
        from os import makedirs
        from os.path import join

        from pandas import DataFrame, read_csv
        from numpy import array, transpose

        # Open original confounds file
        data_frame = read_csv(filepath, sep = '\t', header=0)

        # Extract confounds we want to use for the model
        retained_parameters = DataFrame(transpose(array([
            data_frame['X'], data_frame['Y'], data_frame['Z'],
            data_frame['RotX'], data_frame['RotY'], data_frame['RotZ'],
            data_frame['CSF'], data_frame['WhiteMatter'], data_frame['GlobalSignal']])))

        # Write confounds to a file
        confounds_file = join(working_dir, 'confounds_files',
            f'confounds_file_sub-{subject_id}_run-{run_id}.tsv')

        makedirs(join(working_dir, 'confounds_files'), exist_ok = True)

        with open(confounds_file, 'w', encoding = 'utf-8') as writer:
            writer.write(retained_parameters.to_csv(
                sep = '\t', index = False, header = False, na_rep = '0.0'))

        return confounds_file

    def get_subject_level_analysis(self):
        """
        Create the subject level analysis workflow.

        Returns:
            - subject_level_analysis : nipype.WorkFlow
        """
        # Infosource Node - To iterate on subjects
        information_source = Node(IdentityInterface(
            fields = ['subject_id']),
            name = 'information_source')
        information_source.iterables = [('subject_id', self.subject_list)]

        # Templates to select files node
        template = {
            'confounds' : join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-*_bold_confounds.tsv'),
            'func' : join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-*_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'),
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

        # Smooth - smoothing node
        smoothing = Node(Smooth(), name = 'smoothing')
        smoothing.inputs.fwhm = self.fwhm

        # Function node get_subject_information - get subject specific condition information
        subject_information = Node(Function(
            function = self.get_subject_information,
            input_names = ['event_files'],
            output_names = ['subject_info']),
            name = 'subject_information')

        # SpecifyModel - generates SPM-specific Model
        specify_model = Node(SpecifySPMModel(), name = 'specify_model')
        specify_model.inputs.concatenate_runs = True
        specify_model.inputs.input_units = 'secs'
        specify_model.inputs.output_units = 'secs'
        specify_model.inputs.time_repetition = TaskInformation()['RepetitionTime']
        specify_model.inputs.high_pass_filter_cutoff = 128

        # Level1Design - Generates an SPM design matrix
        model_design = Node(Level1Design(), name = 'model_design')
        model_design.inputs.bases = {'hrf': {'derivs': [0, 0]}}
        model_design.inputs.timing_units = 'secs'
        model_design.inputs.interscan_interval = TaskInformation()['RepetitionTime']

        # EstimateModel - estimate the parameters of the model
        model_estimate = Node(EstimateModel(), name = 'model_estimate')
        model_estimate.inputs.estimation_method = {'Classical': 1}

        # Function node get_confounds_file - get confounds files
        confounds = MapNode(Function(
            function = self.get_confounds_file,
            input_names = ['filepath', 'subject_id', 'run_id', 'working_dir'],
            output_names = ['confounds_file']),
            name = 'confounds', iterfield = ['filepath', 'run_id'])
        confounds.inputs.working_dir = self.directories.working_dir
        confounds.inputs.run_id = self.run_list

        # EstimateContrast - estimates contrasts
        contrast_estimate = Node(EstimateContrast(), name = 'contrast_estimate')
        contrast_estimate.inputs.contrasts = self.subject_level_contrasts

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
            base_dir = self.directories.working_dir,
            name = 'subject_level_analysis'
            )
        subject_level_analysis.connect([
            (information_source, select_files, [('subject_id', 'subject_id')]),
            (subject_information, specify_model, [('subject_info', 'subject_info')]),
            (select_files, confounds, [('confounds', 'filepath')]),
            (select_files, subject_information, [('event', 'event_files')]),
            (information_source, confounds, [('subject_id', 'subject_id')]),
            (select_files, gunzip_func, [('func', 'in_file')]),
            (gunzip_func, smoothing, [('out_file', 'in_files')]),
            (gunzip_func, remove_gunzip_files, [('out_file', 'file_name')]),
            (smoothing, remove_gunzip_files, [('smoothed_files', '_')]),
            (smoothing, remove_smoothed_files, [('smoothed_files', 'file_name')]),
            (smoothing, specify_model, [('smoothed_files', 'functional_runs')]),
            (specify_model, remove_smoothed_files, [('session_info', '_')]),
            (confounds, specify_model, [('confounds_file', 'realignment_parameters')]),
            (specify_model, model_design, [('session_info', 'session_info')]),
            (model_design, model_estimate, [('spm_mat_file', 'spm_mat_file')]),
            (model_estimate, contrast_estimate, [
                ('spm_mat_file', 'spm_mat_file'),
                ('beta_images', 'beta_images'),
                ('residual_image', 'residual_image')]),
            (contrast_estimate, data_sink, [
                ('con_images', 'subject_level_analysis.@con_images'),
                ('spmT_images', 'subject_level_analysis.@spmT_images'),
                ('spm_mat_file', 'subject_level_analysis.@spm_mat_file')
            ])
        ])

        return subject_level_analysis

    def get_subject_level_outputs(self):
        """ Return the names of the files the subject level analysis is supposed to generate. """

        # Contrat maps
        templates = [join(
            self.directories.output_dir,
            'subject_level_analysis', '_subject_id_{subject_id}', f'con_{contrast_id}.nii')\
            for contrast_id in self.contrast_list]

        # SPM.mat file
        templates += [join(
            self.directories.output_dir,
            'subject_level_analysis', '_subject_id_{subject_id}', 'SPM.mat')]

        # spmT maps
        templates += [join(
            self.directories.output_dir,
            'subject_level_analysis', '_subject_id_{subject_id}', f'spmT_{contrast_id}.nii')\
            for contrast_id in self.contrast_list]

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
        # Compute the number of participants used to do the analysis
        nb_subjects = len(self.subject_list)

        # Infosource - a function free node to iterate over the list of subject names
        information_source = Node(
            IdentityInterface(
                fields=['contrast_id']),
                name='information_source')
        information_source.iterables = [('contrast_id', self.contrast_list)]

        # SelectFiles
        templates = {
            # Contrasts for all participants
            'contrasts' : join(self.directories.output_dir,
                'subject_level_analysis', '_subject_id_*', 'con_{contrast_id}.nii')
        }

        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.results_dir
        select_files.inputs.force_lists = True

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

        ## Create thresholded maps
        threshold = MapNode(Threshold(),
            name = 'threshold',
            iterfield = ['stat_image', 'contrast_index'])
        threshold.inputs.use_fwe_correction = False
        threshold.inputs.height_threshold = 0.001
        threshold.inputs.extent_fdr_p_threshold = 0.05
        threshold.inputs.use_topo_fdr = False
        threshold.inputs.force_activation = True
        threshold.synchronize = True

        group_level_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = f'group_level_analysis_{method}_nsub_{nb_subjects}')
        group_level_analysis.connect([
            (information_source, select_files, [('contrast_id', 'contrast_id')]),
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
            two_sample_t_test_design.inputs.unequal_variance = True

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
        parameters = {
            'contrast_id': self.contrast_list,
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
            '_contrast_id_{contrast_id}',
            '{file}'
            )

        return_list = [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

        # Handle groupComp
        parameters = {
            'contrast_id': self.contrast_list,
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
            '_contrast_id_{contrast_id}',
            '{file}'
            )

        return_list += [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

        return return_list

    def get_hypotheses_outputs(self):
        """ Return all hypotheses output file names.
            Note that hypotheses 5 to 8 correspond to the maps given by the team in their results ;
            but they are not fully consistent with the hypotheses definitions as expected by NARPS.
        """
        nb_sub = len(self.subject_list)
        files = [
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0001.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0001.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0001.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0001.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0003', '_threshold1', 'spmT_0002_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0003', 'spmT_0002.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0003', '_threshold1', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0003', 'spmT_0001.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0003', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0003', 'spmT_0001.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0003', '_threshold0', 'spmT_0002_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0003', 'spmT_0002.nii'),
            join(f'group_level_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_0003', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_0003', 'spmT_0001.nii')
        ]
        return [join(self.directories.output_dir, f) for f in files]
