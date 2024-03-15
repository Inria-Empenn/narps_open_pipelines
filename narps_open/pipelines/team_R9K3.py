#!/usr/bin/python
# coding: utf-8

""" Write the work of NARPS' team R9K3 using Nipype """

from os.path import join
from itertools import product

from nipype import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.utility.base import Merge
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
from narps_open.core.interfaces import InterfaceFactory
from narps_open.core.common import (
    list_intersection, elements_in_string, clean_list
    )
from narps_open.utils.configuration import Configuration

class PipelineTeamR9K3(Pipeline):
    """ A class that defines the pipeline of team R9K3. """

    def __init__(self):
        super().__init__()
        self.fwhm = 6.0
        self.team_id = 'R9K3'
        self.contrast_list = ['0001', '0002']
        conditions = ['trialxgain^1', 'trialxloss^1']
        self.subject_level_contrasts = [
            ['effect_of_gain', 'T', conditions, [1, 0]],
            ['effect_of_loss', 'T', conditions, [0, 1]]
            ]

    def get_preprocessing(self):
        """
        Create the preprocessing workflow.

        Returns:
            - preprocessing : nipype.WorkFlow
        """
        # Initialize preprocessing workflow to connect nodes along the way
        preprocessing = Workflow(
            base_dir = self.directories.working_dir,
            name = 'preprocessing')

        # IDENTITY INTERFACE - To iterate on subjects
        information_source = Node(IdentityInterface(
            fields = ['subject_id']),
            name = 'information_source')
        information_source.iterables = [('subject_id', self.subject_list)]

        # SELECT FILES - to select necessary files
        templates = {
            'func' : join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-*_bold_space-MNI152NLin2009cAsym_preproc.nii.gz')
        }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir
        preprocessing.connect(information_source, 'subject_id', select_files, 'subject_id')

        # GUNZIP - gunzip files because SPM do not use .nii.gz files
        gunzip_func = MapNode(Gunzip(), name = 'gunzip_func', iterfield = ['in_file'])
        preprocessing.connect(select_files, 'func', gunzip_func, 'in_file')

        # SMOOTH - Spatial smoothing of fMRI data.
        smoothing = MapNode(Smooth(), name = 'smoothing', iterfield = 'in_files')
        smoothing.inputs.fwhm = [self.fwhm] * 3
        preprocessing.connect(gunzip_func, 'out_file', smoothing, 'in_files')

        # DATA SINK - store the wanted results in the wanted repository
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir
        preprocessing.connect(smoothing, 'smoothed_files', data_sink, 'preprocessing.@smoothed')

        # Remove large files, if requested
        if Configuration()['pipelines']['remove_unused_data']:

            # MERGE - Merge all temporary outputs once they are no longer needed
            merge_temp_files = Node(Merge(2), name = 'merge_temp_files')
            preprocessing.connect(gunzip_func, 'out_file', merge_temp_files, 'in1')
            preprocessing.connect(smoothing, 'smoothed_files', merge_temp_files, 'in2')

            # FUNCTION - Remove temporary files once they are no longer needed
            remove_gunziped = MapNode(
                InterfaceFactory.create('remove_parent_directory'),
                name = 'remove_gunziped',
                iterfield = 'file_name'
                )
            preprocessing.connect(merge_temp_files, 'out', remove_gunziped, 'file_name')
            preprocessing.connect(data_sink, 'out_file', remove_gunziped, '_')

        return preprocessing

    def get_preprocessing_outputs(self):
        """ Return the names of the files the preprocessing is supposed to generate. """

        output_dir = join(self.directories.output_dir, 'preprocessing', '_subject_id_{subject_id}')

        # Smoothing outputs
        templates = [join(output_dir, f'_smoothing{index}',
            'ssub-{subject_id}'+f'_task-MGT_run-{run_id}_bold.nii')\
            for index, run_id in zip(range(len(self.run_list)), self.run_list)]

        # Format with subject_ids
        return_list = []
        for template in templates:
            return_list += [template.format(subject_id = s) for s in self.subject_list]

        return return_list

    def get_run_level_analysis(self):
        """ No run level analysis has been done by team R9K3 """
        return None

    def get_subject_information(event_file):
        """
        Create Bunchs for specifySPMModel, from data extracted from an event_file.

        Parameters :
        - event_files: str, event file (one per run) for the subject

        Returns :
        - subject_information: Bunch, relevant event information for subject level analysis.
        """
        from nipype.interfaces.base import Bunch
        from numpy import max as npmax

        trial_onsets = []
        trial_durations = []
        weights_gain = []
        weights_loss = []

        with open(event_file, 'rt') as file:
            next(file)  # skip the header

            for line in file:
                info = line.strip().split()

                trial_onsets.append(float(info[0]))
                trial_durations.append(0.0)
                weights_gain.append(float(info[2]))
                weights_loss.append(float(info[3]))

        # Scale weights between 0 and 1
        weights_gain = list(weights_gain / npmax(weights_gain))
        weights_loss = list(weights_loss / npmax(weights_loss))

        return Bunch(
            conditions = ['trial'],
            onsets = [trial_onsets],
            durations = [trial_durations],
            amplitudes = None,
            tmod = None,
            pmod = [
                Bunch(
                    name = ['gain', 'loss'],
                    poly = [1, 1],
                    param = [weights_gain, weights_loss],
                    ),
                    None,
                ],
                regressor_names = None,
                regressors = None
            )

    def get_confounds_file(confounds_file: str, subject_id: str, run_id: str) -> str:
        """
        Create a tsv file with only desired confounds per subject per run.

        Parameters :
        - confounds_file: str, path to the file containing confounds from fmriprep
        - subject_id : related subject id
        - run_id : related run id

        Return :
        - out_file : path to new file containing only desired confounds
        """
        from os.path import abspath

        from pandas import read_csv, DataFrame
        from numpy import array, transpose

        # Get the dataframe containing the 6 head motion parameter regressors
        data_frame = read_csv(confounds_file, sep = '\t', header=0)

        # Extract parameters we want to use for the model
        retained_parameters = DataFrame(transpose(array([
            data_frame['X'], data_frame['Y'], data_frame['Z'],
            data_frame['RotX'], data_frame['RotY'], data_frame['RotZ']])))

        # Write confounds to a file
        out_file = abspath(f'confounds_file_sub-{subject_id}_run-{run_id}.tsv')
        with open(out_file, 'w', encoding = 'utf-8') as writer:
            writer.write(retained_parameters.to_csv(
                sep = '\t', index = False, header = False, na_rep = '0.0'))

        return out_file

    def get_subject_level_analysis(self):
        """
        Create the subject level analysis workflow.

        Returns:
            - subject_level : nipype.WorkFlow
                WARNING: the name attribute of the workflow is 'subject_level_analysis'
        """
        # Create subject level analysis workflow
        subject_level = Workflow(
            base_dir = self.directories.working_dir,
            name = 'subject_level_analysis')

        # IDENTITY INTERFACE - To iterate on subjects
        information_source = Node(IdentityInterface(
            fields = ['subject_id']),
            name = 'information_source')
        information_source.iterables = [('subject_id', self.subject_list)]

        # SELECT FILES - to select necessary files
        templates = {
            'confounds' : join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-*_bold_confounds.tsv'),
            'func' : join(self.directories.output_dir, 'preprocessing', '_subject_id_{subject_id}',
                '_smoothing*',
                'ssub-{subject_id}_task-MGT_run-*_bold_space-MNI152NLin2009cAsym_preproc.nii'),
            'event' : join('sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-*_events.tsv')
        }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir
        subject_level.connect(information_source, 'subject_id', select_files, 'subject_id')

        # FUNCTION get_subject_information - generate files with event data
        subject_information = MapNode(Function(
            function = self.get_subject_information,
            input_names = ['event_file'],
            output_names = ['subject_info']),
            iterfield = 'event_file',
            name = 'subject_information')
        subject_level.connect(select_files, 'event', subject_information, 'event_file')

        # FUNCTION node get_confounds_file - generate files with confounds data
        confounds = MapNode(
            Function(
                function = self.get_confounds_file,
                input_names = ['confounds_file', 'subject_id', 'run_id'],
                output_names = ['confounds_file']
            ),
            name = 'confounds',
            iterfield = ['confounds_file', 'run_id'])
        confounds.inputs.run_id = self.run_list
        subject_level.connect(information_source, 'subject_id', confounds, 'subject_id')
        subject_level.connect(select_files, 'confounds', confounds, 'confounds_file')

        # SPECIFY MODEL - generates SPM-specific Model
        specify_model = Node(SpecifySPMModel(), name = 'specify_model')
        specify_model.inputs.input_units = 'secs'
        specify_model.inputs.output_units = 'secs'
        specify_model.inputs.time_repetition = TaskInformation()['RepetitionTime']
        specify_model.inputs.high_pass_filter_cutoff = 128
        subject_level.connect(select_files, 'func', specify_model, 'functional_runs')
        subject_level.connect(confounds, 'confounds_file', specify_model, 'realignment_parameters')
        subject_level.connect(subject_information, 'subject_info', specify_model, 'subject_info')

        # LEVEL 1 DESIGN - Generates an SPM design matrix
        model_design = Node(Level1Design(), name = 'model_design')
        model_design.inputs.bases = {'hrf': {'derivs': [0, 0]}}
        model_design.inputs.timing_units = 'secs'
        model_design.inputs.interscan_interval = TaskInformation()['RepetitionTime']
        subject_level.connect(specify_model, 'session_info', model_design, 'session_info')

        # ESTIMATE MODEL - estimate the parameters of the model
        model_estimate = Node(EstimateModel(), name = 'model_estimate')
        model_estimate.inputs.estimation_method = {'Classical': 1}
        subject_level.connect(model_design, 'spm_mat_file', model_estimate, 'spm_mat_file')

        # ESTIMATE CONTRAST - estimates contrasts
        contrast_estimate = Node(EstimateContrast(), name = 'contrast_estimate')
        contrast_estimate.inputs.contrasts = self.subject_level_contrasts
        subject_level.connect([
            (model_estimate, contrast_estimate, [
                ('spm_mat_file', 'spm_mat_file'),
                ('beta_images', 'beta_images'),
                ('residual_image', 'residual_image')
            ])
        ])

        # DATA SINK - store the wanted results in the wanted repository
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir
        subject_level.connect([
            (contrast_estimate, data_sink, [
                ('con_images', 'subject_level_analysis.@con_images'),
                ('spmT_images', 'subject_level_analysis.@spmT_images'),
                ('spm_mat_file', 'subject_level_analysis.@spm_mat_file')
            ])
        ])

        return subject_level

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
        threshold.inputs.extent_threshold = 5
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
                '_contrast_id_0001', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001', 'spmT_0001.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001', 'spmT_0001.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001', 'spmT_0001.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001', 'spmT_0001.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold1', 'spmT_0002_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0002.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold1', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0001.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0001.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold0', 'spmT_0002_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0002.nii'),
            join(f'group_level_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0001.nii')
        ]
        return [join(self.directories.output_dir, f) for f in files]
