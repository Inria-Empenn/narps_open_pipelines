#!/usr/bin/python
# coding: utf-8

""" Write the work of NARPS team 51PW using Nipype """

from os.path import join
from itertools import product

from nipype import Node, Workflow, MapNode
from nipype.interfaces.utility import IdentityInterface, Function, Split
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.fsl import (
    # General usage
    FSLCommand, ImageStats,
    # Preprocessing
    SUSAN,
    # Analyses
    Level1Design, FEATModel, L2Model, FILMGLS,
    FLAMEO, Randomise, MultipleRegressDesign
    )
from nipype.interfaces.fsl.utils import ExtractROI, Merge as MergeImages
from nipype.interfaces.fsl.maths import MathsCommand, MultiImageMaths
from nipype.algorithms.modelgen import SpecifyModel

from narps_open.pipelines import Pipeline
from narps_open.data.task import TaskInformation
from narps_open.data.participants import get_group
from narps_open.core.common import (
    remove_file, list_intersection, elements_in_string, clean_list
    )
from narps_open.core.interfaces import InterfaceFactory
from narps_open.utils.configuration import Configuration

# Setup FSL
FSLCommand.set_default_output_type('NIFTI_GZ')

class PipelineTeam51PW(Pipeline):
    """ A class that defines the pipeline of team 51PW """

    def __init__(self):
        super().__init__()
        self.fwhm = 5.0
        self.team_id = '51PW'
        self.contrast_list = ['1', '2']
        self.run_level_contasts = [
            ('effect_gain', 'T', ['gamble', 'gain', 'loss'], [0, 1, 0]),
            ('effect_loss', 'T', ['gamble', 'gain', 'loss'], [0, 0, 1])
        ]

    def get_preprocessing(self):
        """ Return a Nipype workflow describing the preprocessing part of the pipeline

        Returns:
            - preprocessing : nipype.WorkFlow
        """
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
            # Functional MRI - from the fmriprep derivatives
            'func' : join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'
                ),
            # Mask - from the fmriprep derivatives
            'mask' : join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz'
                )
        }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir

        # DataSink Node - store the wanted results in the wanted directory
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir

        # ImageStats Node - Compute mean value of the 4D data
        #   -k option adds a mask
        #   -m computes the mean value
        #   we do not need to filter on not-zero values (option -P) because a mask is passed
        #   Warning : these options must be passed in the right order
        #       (i.e.: apply mask then compute stat)
        compute_mean = Node(ImageStats(), name = 'compute_mean')
        compute_mean.inputs.op_string = '-k %s -m'

        # MathsCommand Node - Perform grand-mean intensity normalisation of the entire 4D data
        intensity_normalization = Node(MathsCommand(), name = 'intensity_normalization')
        build_ing_args = lambda x : f'-ing {x}'

        # ImageStats Node - Compute median of voxel values to derive SUSAN's brightness_threshold
        #   -k option adds a mask
        #   -p computes the 50th percentile (= median)
        #   we do not need to filter on not-zero values (option -P) because a mask is passed
        #   Warning : these options must be passed in the right order
        #       (i.e.: apply mask then compute stat)
        compute_median = Node(ImageStats(), name = 'compute_median')
        compute_median.inputs.op_string = '-k %s -p 50'

        # SUSAN Node - smoothing of functional images
        #   we set brightness_threshold to .75x median of the input file, as performed by fMRIprep
        smoothing = Node(SUSAN(), name = 'smoothing')
        smoothing.inputs.fwhm = self.fwhm
        compute_brightness_threshold = lambda x : .75 * x

        # Define workflow
        preprocessing = Workflow(base_dir = self.directories.working_dir, name = 'preprocessing')
        preprocessing.config['execution']['stop_on_first_crash'] = 'true'
        preprocessing.connect([
            (information_source, select_files, [
                ('subject_id', 'subject_id'), ('run_id', 'run_id')
                ]),
            (select_files, compute_mean, [('func', 'in_file')]),
            (select_files, compute_mean, [('mask', 'mask_file')]),
            (select_files, intensity_normalization, [('func', 'in_file')]),
            (select_files, compute_median, [('func', 'in_file')]),
            (select_files, compute_median, [('mask', 'mask_file')]),
            (compute_mean, intensity_normalization, [
                (('out_stat', build_ing_args), 'args')
                ]),
            (compute_median, smoothing, [
                (('out_stat', compute_brightness_threshold), 'brightness_threshold')
                ]),
            (intensity_normalization, smoothing, [('out_file', 'in_file')]),
            (smoothing, data_sink, [('smoothed_file', 'preprocessing.@output_image')])
            ])

        # Remove large files, if requested
        if Configuration()['pipelines']['remove_unused_data']:

            # Remove Node - Remove intensity normalization files once they are no longer needed
            remove_intensity_normalization = Node(
                InterfaceFactory.create('remove_parent_directory'),
                name = 'remove_intensity_normalization'
                )

            # Remove Node - Remove smoothed files once they are no longer needed
            remove_smooth = Node(
                InterfaceFactory.create('remove_parent_directory'),
                name = 'remove_smooth'
                )

            # Add connections
            preprocessing.connect([
                (data_sink, remove_intensity_normalization, [('out_file', '_')]),
                (intensity_normalization, remove_intensity_normalization, [('out_file', 'file_name')]),
                (data_sink, remove_smooth, [('out_file', '_')]),
                (smoothing, remove_smooth, [('smoothed_file', 'file_name')]),
                ])

        return preprocessing

    def get_preprocessing_outputs(self):
        """ Return a list of the files generated by the preprocessing """

        parameters = {
            'subject_id': self.subject_list,
            'run_id': self.run_list,
        }
        parameter_sets = product(*parameters.values())
        template = join(
            self.directories.output_dir,
            'preprocessing',
            '_run_id_{run_id}_subject_id_{subject_id}',
            'sub-{subject_id}_task-MGT_run-{run_id}_bold_space-MNI152NLin2009cAsym_preproc_maths_smooth.nii.gz'
            )

        return [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

    def get_subject_information(event_file):
        """
        Extract information from an event file, to setup the model.

        Parameters :
        - event_file : str, event file corresponding to the run and the subject to analyze

        Returns :
        - subject_info : list of Bunch containing event information
        """
        from nipype.interfaces.base import Bunch

        condition_names = ['gamble', 'gain', 'loss']
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
                onsets['gamble'].append(float(info[0]))
                durations['gamble'].append(float(info[1]))
                amplitudes['gamble'].append(1.0)
                onsets['gain'].append(float(info[0]))
                durations['gain'].append(float(info[1]))
                amplitudes['gain'].append(float(info[2]))
                onsets['loss'].append(float(info[0]))
                durations['loss'].append(float(info[1]))
                amplitudes['loss'].append(float(info[3]))

        return [
            Bunch(
                conditions = condition_names,
                onsets = [onsets[k] for k in condition_names],
                durations = [durations[k] for k in condition_names],
                amplitudes = [amplitudes[k] for k in condition_names],
                regressor_names = None,
                regressors = None)
            ]

    def get_confounds(in_file, subject_id, run_id, working_dir):
        """
        Create a new tsv file with only desired confounds for one subject, one run.

        Parameters :
        - in_file : paths to subject's and run's confounds file
        - subject_id : subject id to pick the file from
        - run_id : run id to pick the file from

        Return :
        - path to a new file containing only desired confounds.
        """
        from os import makedirs
        from os.path import join

        from pandas import read_csv, DataFrame
        from numpy import array, transpose

        # Read input confounds file
        data_frame = read_csv(in_file, sep = '\t', header=0)

        # Extract confounds we want to use for the model
        confounds = DataFrame(transpose(array([
            data_frame['X'], data_frame['Y'], data_frame['Z'],
            data_frame['RotX'], data_frame['RotY'], data_frame['RotZ'],
            data_frame['FramewiseDisplacement'], data_frame['aCompCor00'],
            data_frame['aCompCor01'], data_frame['aCompCor02'],
            data_frame['aCompCor03'], data_frame['aCompCor04']
            ])))

        # Exclude 2 time points for each run
        confounds = confounds.iloc[2:]

        # Write confounds to a file
        confounds_file_path = join(working_dir, 'confounds_files',
            f'confounds_file_sub-{subject_id}_run-{run_id}.tsv')

        makedirs(join(working_dir, 'confounds_files'), exist_ok = True)

        with open(confounds_file_path, 'w') as writer:
            writer.write(confounds.to_csv(
                    sep = '\t', index = False, header = False, na_rep = '0.0'))

        return confounds_file_path

    def get_run_level_analysis(self):
        """ Return a Nipype workflow describing the run level analysis part of the pipeline

        Returns:
            - run_level_analysis : nipype.WorkFlow
        """
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
            # Functional MRI - from the preprocessing
            'func' : join(self.directories.output_dir, 'preprocessing',
                '_run_id_{run_id}_subject_id_{subject_id}',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_space-MNI152NLin2009cAsym_preproc_maths_smooth.nii.gz'
                ),
            # Events file - from the original dataset
            'events' : join('sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_events.tsv'
                ),
            # Confounds values - from the fmriprep derivatives
            'confounds' : join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_confounds.tsv'
                )
        }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir

        # DataSink Node - store the wanted results in the wanted directory
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir

        # ExtractROI Node - Exclude the first 2 time points/scans for each run
        exclude_time_points = Node(ExtractROI(), name = 'exclude_time_points')
        exclude_time_points.inputs.t_min = 2
        exclude_time_points.inputs.t_size = 451 # TODO nb_time_points - 2

        # Function Node get_subject_information - Get subject information from event files
        subject_information = Node(Function(
            function = self.get_subject_information,
            input_names = ['event_file'],
            output_names = ['subject_info']
            ), name = 'subject_information')

        # Function MapNode get_confounds - Get subject information from event files
        select_confounds = Node(Function(
            function = self.get_confounds,
            input_names = ['in_file', 'subject_id', 'run_id', 'working_dir'],
            output_names = ['confounds_file']
            ), name = 'select_confounds', iterfield = ['run_id'])
        select_confounds.inputs.working_dir = self.directories.working_dir

        # SpecifyModel Node - Generates a model
        specify_model = Node(SpecifyModel(), name = 'specify_model')
        specify_model.inputs.high_pass_filter_cutoff = 50.0
        specify_model.inputs.input_units = 'secs'
        specify_model.inputs.time_repetition = TaskInformation()['RepetitionTime']

        # Level1Design Node - Generate files for first level computation
        model_design = Node(Level1Design(), 'model_design')
        model_design.inputs.bases = {
            'gamma':{'derivs' : False} # Canonical gamma HRF
            }
        model_design.inputs.interscan_interval = TaskInformation()['RepetitionTime']
        model_design.inputs.model_serial_correlations = True
        model_design.inputs.contrasts = self.run_level_contasts

        # FEATModel Node - Generate first level model
        model_generation = Node(FEATModel(), name = 'model_generation')

        # FILMGLS Node - Estimate first level model
        model_estimate = Node(FILMGLS(), name = 'model_estimate')
        #model_estimate.inputs.fit_armodel = True
        #The noise level was set to the default 0.66.
        #The AR(1) parameter was fixed to 0.34.

        # Create l1 analysis workflow and connect its nodes
        run_level_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = 'run_level_analysis'
            )
        run_level_analysis.connect([
            (information_source, select_files, [
                ('subject_id', 'subject_id'), ('run_id', 'run_id')
                ]),
            (information_source, select_confounds, [
                ('subject_id', 'subject_id'), ('run_id', 'run_id')
                ]),
            (select_files, subject_information, [('events', 'event_file')]),
            (select_files, select_confounds, [('confounds', 'in_file')]),
            (select_files, exclude_time_points, [('func', 'in_file')]),
            (exclude_time_points, specify_model, [('roi_file', 'functional_runs')]),
            (exclude_time_points, model_estimate, [('roi_file', 'in_file')]),
            (subject_information, specify_model, [('subject_info', 'subject_info')]),
            (select_confounds, specify_model, [('confounds_file', 'realignment_parameters')]),
            (specify_model, model_design, [('session_info', 'session_info')]),
            (model_design, model_generation, [
                ('ev_files', 'ev_files'),
                ('fsf_files', 'fsf_file')]),
            (model_generation, model_estimate, [
                ('con_file', 'tcon_file'),
                ('design_file', 'design_file')]),
            (model_estimate, data_sink, [('results_dir', 'run_level_analysis.@results')]),
            (model_generation, data_sink, [
                ('design_file', 'run_level_analysis.@design_file'),
                ('design_image', 'run_level_analysis.@design_img')]),
            ])

        # Remove large files, if requested
        if Configuration()['pipelines']['remove_unused_data']:

            # Remove Node - Remove smoothed files (computed by preprocessing)
            #   once they are no longer needed
            remove_smooth = MapNode(
                InterfaceFactory.create('remove_file'),
                name = 'remove_smooth',
                iterfield = ['file_name']
                )

            # Remove Node - Remove roi files once they are no longer needed
            remove_roi = Node(
                InterfaceFactory.create('remove_parent_directory'),
                name = 'remove_roi'
                )

            # Add connections
            run_level_analysis.connect([
                (data_sink, remove_smooth, [('out_file', '_')]),
                (select_files, remove_smooth, [('func', 'file_name')]),
                (data_sink, remove_roi, [('out_file', '_')]),
                (exclude_time_points, remove_roi, [('roi_file', 'file_name')])
                ])

        return run_level_analysis

    def get_run_level_outputs(self):
        """ Return a list of the files generated by the run level analysis """

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
        """ Return a Nipype workflow describing the subject level analysis part of the pipeline """

        # IdentityInterface node - allows to iterate over subjects and contrasts
        information_source = Node(IdentityInterface(
            fields = ['subject_id', 'contrast_id']),
            name = 'information_source')
        information_source.iterables = [
            ('subject_id', self.subject_list),
            ('contrast_id', self.contrast_list)
            ]

        # SelectFiles Node - select necessary files
        templates = {
            'copes' : join(self.directories.output_dir, 'run_level_analysis',
                '_run_id_*_subject_id_{subject_id}', 'results', 'cope{contrast_id}.nii.gz'),
            'varcopes' : join(self.directories.output_dir, 'run_level_analysis',
                '_run_id_*_subject_id_{subject_id}', 'results', 'varcope{contrast_id}.nii.gz'),
            'masks' : join(self.directories.output_dir, 'preprocessing',
                '_run_id_*_subject_id_{subject_id}',
                'sub-{subject_id}_task-MGT_run-*_bold_brain_mask_flirt_wtsimt.nii.gz')
        }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir

        # DataSink Node - store the wanted results in the wanted directory
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir

        # L2Model Node - Generate subject specific second level model
        generate_model = Node(L2Model(), name = 'generate_model')
        generate_model.inputs.num_copes = len(self.run_list)

        # Merge Node - Merge copes files for each subject
        merge_copes = Node(MergeImages(), name = 'merge_copes')
        merge_copes.inputs.dimension = 't'

        # Merge Node - Merge varcopes files for each subject
        merge_varcopes = Node(MergeImages(), name = 'merge_varcopes')
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
        estimate_model.inputs.run_mode = 'fe' # Fixed effect

        # Second level (single-subject, mean of all four scans) analyses: Fixed effects analysis.
        subject_level_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = 'subject_level_analysis')
        subject_level_analysis.connect([
            (information_source, select_files, [
                ('subject_id', 'subject_id'),
                ('contrast_id', 'contrast_id')]),
            (select_files, merge_copes, [('copes', 'in_files')]),
            (select_files, merge_varcopes, [('varcopes', 'in_files')]),
            (select_files, split_masks, [('masks', 'inlist')]),
            (split_masks, mask_intersection, [('out1', 'in_file')]),
            (split_masks, mask_intersection, [('out2', 'operand_files')]),
            (merge_copes, estimate_model, [('merged_file', 'cope_file')]),
            (merge_varcopes, estimate_model, [('merged_file', 'var_cope_file')]),
            (mask_intersection, estimate_model, [('out_file', 'mask_file')]),
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
        """ Return a list of the files generated by the subject level analysis """

        parameters = {
            'contrast_id' : self.contrast_list,
            'subject_id' : self.subject_list,
            'file' : ['cope1.nii.gz', 'tstat1.nii.gz', 'varcope1.nii.gz', 'zstat1.nii.gz']
        }
        parameter_sets = product(*parameters.values())
        template = join(
            self.directories.output_dir,
            'subject_level_analysis', '_contrast_id_{contrast_id}_subject_id_{subject_id}','{file}'
            )

        return [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

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
        """ Return all workflows for the group level analysis. """

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
        information_source = Node(
            IdentityInterface(
                fields = ['contrast_id']
            ),
            name = 'information_source',
        )
        information_source.iterables = [('contrast_id', self.contrast_list)]

        # SelectFiles Node - select necessary files
        templates = {
            'copes' : join(self.directories.output_dir, 'subject_level_analysis',
                '_contrast_id_{contrast_id}_subject_id_*', 'cope1.nii.gz'),
            'varcopes' : join(self.directories.output_dir, 'subject_level_analysis',
                '_contrast_id_{contrast_id}_subject_id_*', 'varcope1.nii.gz'),
            'masks' : join(self.directories.output_dir, 'subject_level_analysis',
                '_contrast_id_{contrast_id}_subject_id_*',
                'sub-*_task-MGT_run-*_bold_brain_mask_flirt_wtsimt_maths.nii.gz')
        }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir
        select_files.inputs.force_list = True

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
        merge_copes = Node(MergeImages(), name = 'merge_copes')
        merge_copes.inputs.dimension = 't'

        # Merge Node - Merge cope files
        merge_varcopes = Node(MergeImages(), name = 'merge_varcopes')
        merge_varcopes.inputs.dimension = 't'

        # Split Node - Split mask list to serve them as inputs of the MultiImageMaths node.
        split_masks = Node(Split(), name = 'split_masks')
        split_masks.inputs.splits = [1, len(self.subject_list) - 1]
        split_masks.inputs.squeeze = True

        # MultiImageMaths Node - Create a group mask by
        #   computing the intersection of all subject masks.
        mask_intersection = Node(MultiImageMaths(), name = 'mask_intersection')
        mask_intersection.inputs.op_string = '-mul %s ' * (len(self.subject_list) - 1)

        # MultipleRegressDesign Node - Specify model
        specify_model = Node(MultipleRegressDesign(), name = 'specify_model')

        # FLAMEO Node - Estimate model
        estimate_model = Node(FLAMEO(), name = 'estimate_model')
        estimate_model.inputs.run_mode = 'ols' # Ordinary least squares

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
            name = f'group_level_analysis_{method}_nsub_{nb_subjects}'
        )
        group_level_analysis.connect([
            (information_source, select_files, [('contrast_id', 'contrast_id')]),
            (select_files, get_copes, [('copes', 'input_str')]),
            (select_files, get_varcopes, [('varcopes', 'input_str')]),
            (select_files, split_masks, [('masks', 'inlist')]),
            (split_masks, mask_intersection, [('out1', 'in_file')]),
            (split_masks, mask_intersection, [('out2', 'operand_files')]),
            (get_copes, merge_copes, [(('out_list', clean_list), 'in_files')]),
            (get_varcopes, merge_varcopes,[(('out_list', clean_list), 'in_files')]),
            (merge_copes, estimate_model, [('merged_file', 'cope_file')]),
            (merge_varcopes, estimate_model, [('merged_file', 'var_cope_file')]),
            (mask_intersection, estimate_model, [('out_file', 'mask_file')]),
            (specify_model, estimate_model, [
                ('design_mat', 'design_file'),
                ('design_con', 't_con_file'),
                ('design_grp', 'cov_split_file')
                ]),
            (merge_copes, randomise, [('merged_file', 'in_file')]),
            (mask_intersection, randomise, [('out_file', 'mask')]),
            (specify_model, randomise, [
                ('design_mat', 'design_mat'),
                ('design_con', 'tcon')
                ]),
            (randomise, data_sink, [
                ('t_corrected_p_files',
                    f'group_level_analysis_{method}_nsub_{nb_subjects}.@tcorpfile'),
                ('tstat_files', f'group_level_analysis_{method}_nsub_{nb_subjects}.@tstat')
                ]),
            (estimate_model, data_sink, [
                ('zstats', f'group_level_analysis_{method}_nsub_{nb_subjects}.@zstats'),
                ('tstats', f'group_level_analysis_{method}_nsub_{nb_subjects}.@tstats')
                ])
        ])

        if method in ('equalRange', 'equalIndifference'):

            # Setup a one sample t-test
            specify_model.inputs.contrasts = [
                ('Group', 'T', ['group_mean'], [1]),
                ('Group', 'T', ['group_mean'], [-1])
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
            specify_model.inputs.contrasts = [(
                'Eq range vs Eq indiff in loss',
                'T',
                ['equalRange', 'equalIndifference'],
                [1, -1]
                )]

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
        parameters = {
            'contrast_id': self.contrast_list,
            'method' : ['groupComp'],
            'file' : [
                'randomise_tfce_corrp_tstat1.nii.gz',
                'randomise_tstat1.nii.gz',
                'zstat1.nii.gz',
                'tstat1.nii.gz'
                ],
            'nb_subjects' : [str(len(self.subject_list))]
        }
        parameter_sets = product(*parameters.values())
        template = join(
            self.directories.output_dir,
            'group_level_analysis_{method}_nsub_{nb_subjects}',
            '_contrast_id_{contrast_id}', '{file}')

        return_list += [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

        return return_list

    def get_hypotheses_outputs(self):
        """ Return the names of the files used by the team to answer the hypotheses of NARPS. """

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
