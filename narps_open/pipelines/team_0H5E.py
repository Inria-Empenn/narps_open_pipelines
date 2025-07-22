#!/usr/bin/python
# coding: utf-8

""" Write the work of NARPS team 0H5E using Nipype """

from os.path import join
from itertools import product

from nipype import Node, Workflow, MapNode
from nipype.interfaces.utility import IdentityInterface, Function, Merge
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.algorithms.misc import Gunzip
from nipype.algorithms.modelgen import SpecifySPMModel
from nipype.interfaces.spm import (
   Realign, Coregister, Normalize, Smooth,
   Level1Design, OneSampleTTestDesign, TwoSampleTTestDesign,
   EstimateModel, EstimateContrast, Threshold
   )
from nipype.interfaces.spm.base import Info as SPMInfo

from narps_open.pipelines import Pipeline
from narps_open.data.task import TaskInformation
from narps_open.data.participants import get_group
from narps_open.core.common import (
    remove_parent_directory, list_intersection, elements_in_string, clean_list
    )
from narps_open.core.image import get_image_timepoints
from narps_open.utils.configuration import Configuration

class PipelineTeam0H5E(Pipeline):
    """ A class that defines the pipeline of team 0H5E """

    def __init__(self):
        super().__init__()
        self.fwhm = 9.0
        self.team_id = '0H5E'
        self.subject_level_models = ['gainfirst', 'lossfirst']

        # Contrast '0001' corresponds to:
        #  - effect_of_loss in the 'gainfirst' model
        #  - effect_of_gain in the 'lossfirst' model
        self.contrast_list = ['0001']

    def get_preprocessing(self):
        """ Return a Nipype workflow describing the prerpocessing part of the pipeline """

        # Workflow initialization
        preprocessing = Workflow(
            base_dir = self.directories.working_dir,
            name = 'preprocessing'
        )
        # IDENTITY INTERFACE - allows to iterate over subjects
        information_source_subject = Node(IdentityInterface(
            fields = ['subject_id']),
            name = 'information_source_subject'
        )
        information_source_subject.iterables = ('subject_id', self.subject_list)

        # IDENTITY INTERFACE - allows to iterate over runs
        information_source_runs = Node(IdentityInterface(
            fields = ['subject_id', 'run_id']),
            name = 'information_source_runs'
        )
        information_source_runs.iterables = ('run_id', self.run_list)
        preprocessing.connect(
            information_source_subject, 'subject_id', information_source_runs, 'subject_id')

        # SELECT FILES - to select subject files
        file_templates = {'anat': join('sub-{subject_id}', 'anat', 'sub-{subject_id}_T1w.nii.gz')}
        select_subject_files = Node(SelectFiles(file_templates), name = 'select_subject_files')
        select_subject_files.inputs.base_directory = self.directories.dataset_dir
        preprocessing.connect(
            information_source_subject, 'subject_id', select_subject_files, 'subject_id')

        # SELECT FILES - to select run files
        file_templates = {
            'func': join('sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold.nii.gz')
        }
        select_run_files = Node(SelectFiles(file_templates), name = 'select_run_files')
        select_run_files.inputs.base_directory = self.directories.dataset_dir
        preprocessing.connect(
            information_source_runs, 'subject_id', select_run_files, 'subject_id')
        preprocessing.connect(information_source_runs, 'run_id', select_run_files, 'run_id')

        # GUNZIP input files
        gunzip_func = Node(Gunzip(), name = 'gunzip_func')
        gunzip_anat = Node(Gunzip(), name = 'gunzip_anat')
        preprocessing.connect(select_run_files, 'func', gunzip_func, 'in_file')
        preprocessing.connect(select_subject_files, 'anat', gunzip_anat, 'in_file')

        # EXTRACTROI - remove first image of func
        # > Removal of "dummy" scans (deleting the first four volumes from each run)
        remove_first_image = Node(Function(
            function = get_image_timepoints,
            input_names = ['in_file', 'start_time_point', 'end_time_point'],
            output_names = ['roi_file']
            ), name = 'remove_first_image')
        remove_first_image.inputs.start_time_point = 4
        remove_first_image.inputs.end_time_point = 453 # last image
        preprocessing.connect(gunzip_func, 'out_file', remove_first_image, 'in_file')

        # REALIGN - motion correction
        realign = Node(Realign(), name = 'realign')
        # Estimation parameters
        realign.inputs.quality = 1.0
        realign.inputs.fwhm = 5.0 #mm
        realign.inputs.separation = 4.0 #mm
        realign.inputs.register_to_mean = True # 'Register to mean'
        realign.inputs.interp = 7 # '7th Degree B-Spline'
        realign.inputs.wrap = [0, 0, 0] # 'No wrap'
        # Reslicing parameters
        # realign.inputs.write_which = [0, 1] # 'Mean Image Only'
        realign.inputs.write_interp = 7 # '7th Degree B-Spline'
        realign.inputs.write_wrap = [0, 0, 0] # 'No wrap'
        realign.inputs.write_mask = True # 'Mask images'
        preprocessing.connect(remove_first_image, 'roi_file', realign, 'in_files')

        # Get MNI template file from SPM
        mni_template_file = join(SPMInfo.getinfo()['path'], 'toolbox', 'OldNorm', 'T1.nii')

        # COREGISTER - Rigid coregistration of subject anatomical to MNI template
        coregister_anat = Node(Coregister(), name = 'coregister_anat')
        # Estimation parameters
        coregister_anat.inputs.target = mni_template_file
        coregister_anat.inputs.cost_function = 'nmi' # 'Normalized Mututal Information'
        coregister_anat.inputs.separation = [4.0, 2.0]
        coregister_anat.inputs.fwhm = [7.0, 7.0]
        # Reslicing parameters
        coregister_anat.inputs.write_interp = 7 # '7th Degree B-Spline'
        coregister_anat.inputs.write_wrap = [0, 0, 0] # 'No wrap'
        coregister_anat.inputs.write_mask = False # 'Don't mask images'
        preprocessing.connect(gunzip_anat, 'out_file', coregister_anat, 'source')

        # NORMALIZE - Non-rigid registration of subject anatomical to MNI template
        normalize_anat = Node(Normalize(), name = 'normalize_anat')
        # Estimation parameters
        normalize_anat.inputs.template = mni_template_file
        normalize_anat.inputs.source_image_smoothing = 8.0 #mm
        normalize_anat.inputs.affine_regularization_type = 'mni' # 'ICBM space template'
        normalize_anat.inputs.DCT_period_cutoff = 25.0
        normalize_anat.inputs.nonlinear_iterations = 16
        normalize_anat.inputs.nonlinear_regularization = 1.0
        # Write parameters
        normalize_anat.inputs.write_preserve = True # 'preserve concentrations'
        normalize_anat.inputs.write_voxel_sizes = [2.0, 2.0, 2.0] # mm
        normalize_anat.inputs.write_interp = 7 # '7th degree b-spline'
        normalize_anat.inputs.write_wrap = [0, 0, 0] # 'No wrap'
        preprocessing.connect(coregister_anat, 'coregistered_source', normalize_anat, 'source')

        # COREGISTER - Rigid coregistration of functional data to coregistered anatomical image
        coregister_func = Node(Coregister(), name = 'coregister_func')
        # Estimation parameters
        coregister_func.inputs.target = mni_template_file
        coregister_func.inputs.cost_function = 'nmi' # 'Normalized Mututal Information'
        coregister_func.inputs.separation = [4.0, 2.0]
        coregister_func.inputs.fwhm = [7.0, 7.0]
        # Reslicing parameters
        coregister_func.inputs.write_interp = 7 # '7th Degree B-Spline'
        coregister_func.inputs.write_wrap = [0, 0, 0] # 'No wrap'
        coregister_func.inputs.write_mask = False # 'Don't mask images'
        preprocessing.connect(realign, 'mean_image', coregister_func, 'source')
        preprocessing.connect(coregister_anat, 'coregistered_source', coregister_func, 'target')
        preprocessing.connect(realign, 'realigned_files', coregister_func, 'apply_to_files')

        # RESAMPLING - Resample functional image after linear transforms
        #   This step was performed as part of the standard processing stream of the team,
        #   but we don't need its output data here.

        # NORMALIZE - Non-rigid registration of functional data to MNI template
        normalize_func = Node(Normalize(), name = 'normalize_func')
        normalize_func.inputs.jobtype = 'write'
        # Write parameters
        normalize_func.inputs.write_preserve = True # 'preserve concentrations'
        normalize_func.inputs.write_voxel_sizes = [2.5, 2.5, 2.5] # mm
        normalize_func.inputs.write_interp = 7 # '7th degree b-spline'
        normalize_func.inputs.write_wrap = [0, 0, 0] # 'No wrap'
        preprocessing.connect(
            normalize_anat, 'normalization_parameters', normalize_func, 'parameter_file')
        preprocessing.connect(
            coregister_func, 'coregistered_files', normalize_func, 'apply_to_files')

        # SMOOTHING - 9 mm fixed FWHM smoothing in MNI volume
        smoothing = Node(Smooth(), name = 'smoothing')
        smoothing.inputs.fwhm = self.fwhm
        smoothing.inputs.implicit_masking = False
        preprocessing.connect(normalize_func, 'normalized_files', smoothing, 'in_files')

        # DATASINK - store the wanted results in the wanted repository
        data_sink = Node(DataSink(), name='data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir
        preprocessing.connect(
            realign, 'realignment_parameters', data_sink, 'preprocessing.@motion_parameters')
        preprocessing.connect(smoothing, 'smoothed_files', data_sink, 'preprocessing.@smoothed')

        # Remove large files, if requested
        if Configuration()['pipelines']['remove_unused_data']:

            # Merge Node - Merge func file names to be removed after datasink node is performed
            merge_removable_func_files = Node(Merge(6), name = 'merge_removable_func_files')
            merge_removable_func_files.inputs.ravel_inputs = True

            # Function Nodes remove_files - Remove sizeable files once they aren't needed
            remove_func_after_datasink = MapNode(Function(
                function = remove_parent_directory,
                input_names = ['_', 'file_name'],
                output_names = []
                ), name = 'remove_func_after_datasink', iterfield = 'file_name')
            preprocessing.connect(gunzip_func, 'out_file', merge_removable_func_files, 'in1')
            preprocessing.connect(realign, 'realigned_files', merge_removable_func_files, 'in2')
            preprocessing.connect(
                coregister_func, 'coregistered_files', merge_removable_func_files, 'in3')
            preprocessing.connect(
                normalize_func, 'normalized_files', merge_removable_func_files, 'in4')
            preprocessing.connect(smoothing, 'smoothed_files', merge_removable_func_files, 'in5')
            preprocessing.connect(
                remove_first_image, 'roi_file', merge_removable_func_files, 'in6')
            preprocessing.connect(
                merge_removable_func_files, 'out', remove_func_after_datasink, 'file_name')
            preprocessing.connect(data_sink, 'out_file', remove_func_after_datasink, '_')

            # Merge Node - Merge anat file names to be removed after datasink node is performed
            merge_removable_anat_files = Node(Merge(3), name = 'merge_removable_anat_files')
            merge_removable_anat_files.inputs.ravel_inputs = True

            # Function Nodes remove_files - Remove sizeable files once they aren't needed
            remove_anat_after_datasink = MapNode(Function(
                function = remove_parent_directory,
                input_names = ['_', 'file_name'],
                output_names = []
                ), name = 'remove_anat_after_datasink', iterfield = 'file_name')
            preprocessing.connect(gunzip_anat, 'out_file', merge_removable_anat_files, 'in1')
            preprocessing.connect(
                coregister_anat, 'coregistered_source', merge_removable_anat_files, 'in2')
            preprocessing.connect(
                normalize_anat, 'normalized_source', merge_removable_anat_files, 'in3')
            preprocessing.connect(
                merge_removable_anat_files, 'out', remove_anat_after_datasink, 'file_name')
            preprocessing.connect(
                data_sink, 'out_file', remove_anat_after_datasink, '_')

        return preprocessing

    def get_preprocessing_outputs(self):
        """ Return the names of the files the preprocessing analysis is supposed to generate. """

        # Smoothed maps
        templates = [join(
            self.directories.output_dir,
            'preprocessing', '_subject_id_{subject_id}', '_run_id_{run_id}',
            'smwrrsub-{subject_id}_task-MGT_run-{run_id}_bold_roi.nii')]

        # Motion parameters file
        templates += [join(
            self.directories.output_dir,
            'preprocessing', '_subject_id_{subject_id}', '_run_id_{run_id}',
            'rp_sub-{subject_id}_task-MGT_run-{run_id}_bold_roi.txt')]

        # Format with subject_ids
        return_list = []
        for template in templates:
            return_list += [template.format(subject_id = s, run_id = r)\
                for r in self.run_list for s in self.subject_list]

        return return_list

    def get_run_level_analysis(self):
        """ Return a Nipype workflow describing the run level analysis part of the pipeline """
        return None

    def get_subject_information(event_file: str, short_run_id: int, first_pmod: str):
        """
        Create Bunchs of subject event information for specifySPMModel.

        Parameters :
        - event_file: str, events file for a run of a subject
        - short_run_id: str, an identifier for the run corresponding to the event_file
            must be '1' for the first run, '2' for the second run, etc.
        - first_pmod: str, either 'gain' or 'loss'
            if 'gain': output Bunch contains the gain values as first parametric modulator
            else: output Bunch contains the loss values as first parametric modulator

        Returns :
        - subject_info : a Bunch corresponding to the event file
        """
        from nipype.interfaces.base import Bunch

        onsets = []
        durations = []
        weights_gain = []
        weights_loss = []

        # Parse event file
        with open(event_file, 'rt') as file:
            next(file)  # skip the header

            for line in file:
                info = line.strip().split()

                # Remove 4 seconds to onset
                onset = float(info[0]) - 4.0

                if onset > 0.0:
                    onsets.append(onset)
                    durations.append(4.0)
                    weights_gain.append(float(info[2]))
                    weights_loss.append(float(info[3]))

        # Parametric modulators
        if first_pmod == 'gain':
            pmods = Bunch(
                name = ['gain', 'loss'],
                poly = [1, 1],
                param = [weights_gain, weights_loss]
            )
        else:
            pmods = Bunch(
                name = ['loss', 'gain'],
                poly = [1, 1],
                param = [weights_loss, weights_gain]
            )

        # Create bunch
        return Bunch(
            conditions = [f'trial_run{short_run_id}'],
            onsets = [onsets],
            durations = [durations],
            amplitudes = None,
            tmod = None,
            pmod = [pmods],
            regressor_names = None,
            regressors = None
        )

    def get_subject_level_analysis(self):
        """ Return workflows describing the subject level analysis part of the pipeline """

        return [self.get_subject_level_analysis_sub_workflow(m) for m in self.subject_level_models]

    def get_subject_level_analysis_sub_workflow(self, model: str):
        """
        Return a Nipype workflow describing one model of the subject level analysis
        part of the pipeline.

        Parameters:
        - model: str, either 'gainfirst' or 'lossfirst'

        Returns: a nipype.Workflow describing the subject level analysis corresponding to
        the gainfirst model (resp. lossfirst model)
        """

        # Workflow initialization
        subject_level_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = f'subject_level_analysis_{model}'
        )

        # IDENTITY INTERFACE - Allows to iterate on subjects
        information_source = Node(IdentityInterface(fields = ['subject_id']),
            name = 'information_source')
        information_source.iterables = [('subject_id', self.subject_list)]

        # SELECTFILES - to select necessary files
        templates = {
            'func': join(self.directories.output_dir, 'preprocessing',
                '_subject_id_{subject_id}', '_run_id_*',
                'smwrrsub-{subject_id}_task-MGT_run-*_bold_roi.nii'
            ),
            'event': join(self.directories.dataset_dir, 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-*_events.tsv'
            ),
            'parameters': join(self.directories.output_dir, 'preprocessing',
                '_subject_id_{subject_id}', '_run_id_*',
                'rp_sub-{subject_id}_task-MGT_run-*_bold_roi.txt'
            )
        }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir
        subject_level_analysis.connect(information_source, 'subject_id', select_files, 'subject_id')

        # FUNCTION node get_subject_information - get subject specific condition information
        #   for 'gainfirst' model (gain as first parametric modulator)
        subject_information = MapNode(Function(
                function = self.get_subject_information,
                input_names = ['event_file', 'short_run_id', 'first_pmod'],
                output_names = ['subject_info']),
            name = 'subject_information', iterfield = ['event_file', 'short_run_id'])
        subject_information.inputs.short_run_id = list(range(1, len(self.run_list) + 1))
        subject_information.inputs.first_pmod = 'gain' if model == 'gainfirst' else 'loss'
        subject_level_analysis.connect(select_files, 'event', subject_information, 'event_file')

        # SPECIFY MODEL - generates SPM-specific Model
        specify_model = Node(SpecifySPMModel(), name = 'specify_model')
        specify_model.inputs.input_units = 'secs'
        specify_model.inputs.output_units = 'secs'
        specify_model.inputs.time_repetition = TaskInformation()['RepetitionTime']
        specify_model.inputs.high_pass_filter_cutoff = 128
        subject_level_analysis.connect(
            subject_information, 'subject_info', specify_model, 'subject_info')
        subject_level_analysis.connect(select_files, 'func', specify_model, 'functional_runs')
        subject_level_analysis.connect(
            select_files, 'parameters', specify_model, 'realignment_parameters')

        # LEVEL1 DESIGN - generates an SPM design matrix
        model_design = Node(Level1Design(), name = 'model_design')
        model_design.inputs.bases = {'hrf': {'derivs': [0, 0]}}
        model_design.inputs.timing_units = 'secs'
        model_design.inputs.interscan_interval = TaskInformation()['RepetitionTime']
        model_design.inputs.model_serial_correlations = 'AR(1)'
        model_design.inputs.microtime_resolution = 16
        model_design.inputs.microtime_onset = 1.0
        model_design.inputs.volterra_expansion_order = 1 #no
        model_design.inputs.global_intensity_normalization = 'none'
        subject_level_analysis.connect(specify_model, 'session_info', model_design, 'session_info')

        # ESTIMATE MODEL - estimate the parameters of the model
        model_estimate = Node(EstimateModel(), name = 'model_estimate')
        model_estimate.inputs.estimation_method = {'Classical': 1}
        subject_level_analysis.connect(
            model_design, 'spm_mat_file', model_estimate, 'spm_mat_file')

        # Create contrasts
        nb_runs = len(self.run_list)
        if model == 'gainfirst':
            subject_level_contrast = [
                'effect_of_loss', 'T',
                [f'trial_run{r}xloss^1' for r in range(1, nb_runs + 1)],
                [1.0 / nb_runs] * nb_runs
            ]
        else:
            subject_level_contrast = [
                'effect_of_gain', 'T',
                [f'trial_run{r}xgain^1' for r in range(1, nb_runs + 1)],
                [1.0 / nb_runs] * nb_runs
            ]

        # ESTIMATE CONTRAST - estimates contrasts
        contrast_estimate = Node(EstimateContrast(), name = 'contrast_estimate')
        contrast_estimate.inputs.contrasts = [subject_level_contrast]
        subject_level_analysis.connect(
            model_estimate, 'spm_mat_file', contrast_estimate, 'spm_mat_file')
        subject_level_analysis.connect(
            model_estimate, 'beta_images', contrast_estimate, 'beta_images')
        subject_level_analysis.connect(
            model_estimate, 'residual_image', contrast_estimate, 'residual_image')

        # DataSink Node - store the wanted results in the wanted repository
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir
        subject_level_analysis.connect(
            contrast_estimate, 'con_images',
            data_sink, f'subject_level_analysis_{model}.@con_images')
        subject_level_analysis.connect(
            contrast_estimate, 'spmT_images',
            data_sink, f'subject_level_analysis_{model}.@spmT_images')
        subject_level_analysis.connect(
            contrast_estimate, 'spm_mat_file',
            data_sink, f'subject_level_analysis_{model}.@spm_mat_file')

        return subject_level_analysis

    def get_subject_level_outputs(self):
        """ Return the names of the files the subject level analysis is supposed to generate. """

        # Contrat maps
        templates = [join(self.directories.output_dir, f'subject_level_analysis_{m}',
            '_subject_id_{subject_id}', f'con_{c}.nii')\
            for c in self.contrast_list for m in self.subject_level_models]

        # SPM.mat file
        templates += [join(self.directories.output_dir, f'subject_level_analysis_{m}',
            '_subject_id_{subject_id}', 'SPM.mat')\
            for m in self.subject_level_models]

        # spmT maps
        templates += [join(self.directories.output_dir, f'subject_level_analysis_{m}',
            '_subject_id_{subject_id}', f'spmT_{contrast_id}.nii')\
            for contrast_id in self.contrast_list for m in self.subject_level_models]

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

        # Initialize workflow
        group_level_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = f'group_level_analysis_{method}_nsub_{nb_subjects}')

        # IDENTITY INTERFACE - iterate over the list of contrasts
        information_source = Node(
            IdentityInterface(
                fields = ['subjects', 'model']),
                name = 'information_source')
        information_source.iterables = [('model', self.subject_level_models)]

        # SELECT FILES - select contrasts for all subjects
        templates = {
            'contrasts' : join('subject_level_analysis_{model}', '_subject_id_*', 'con_*.nii')
            }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.output_dir
        select_files.inputs.force_list = True
        group_level_analysis.connect(
            information_source, 'model', select_files, 'model')

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
        group_level_analysis.connect(select_files, 'contrasts', get_contrasts, 'input_str')

        # ESTIMATE MODEL - (inputs are set below, depending on the method used)
        estimate_model = Node(EstimateModel(), name = 'estimate_model')
        estimate_model.inputs.estimation_method = {'Classical':1}

        # Estimate contrasts
        estimate_contrast = Node(EstimateContrast(), name = 'estimate_contrast')
        estimate_contrast.inputs.group_contrast = True
        group_level_analysis.connect(
            estimate_model, 'spm_mat_file', estimate_contrast, 'spm_mat_file')
        group_level_analysis.connect(
            estimate_model, 'residual_image', estimate_contrast, 'residual_image')
        group_level_analysis.connect(
            estimate_model, 'beta_images', estimate_contrast, 'beta_images')

        # Create thresholded maps
        threshold = MapNode(Threshold(),
            name = 'threshold', iterfield = ['stat_image', 'contrast_index'])
        threshold.inputs.use_topo_fdr = True
        threshold.inputs.height_threshold_type = 'p-value'
        threshold.inputs.extent_threshold = 5
        group_level_analysis.connect(
            estimate_contrast, 'spm_mat_file', threshold, 'spm_mat_file')
        group_level_analysis.connect(
            estimate_contrast, 'spmT_images', threshold, 'stat_image')

        if method in ('equalRange', 'equalIndifference'):
            estimate_contrast.inputs.contrasts = [
                ('Group', 'T', ['mean'], [1]), ('Group', 'T', ['mean'], [-1])
                ]

            threshold.inputs.contrast_index = [1, 2]
            threshold.synchronize = True

            # Specify design matrix
            one_sample_t_test_design = Node(OneSampleTTestDesign(),
                name = 'one_sample_t_test_design')
            one_sample_t_test_design.inputs.use_implicit_threshold = True #implicit masking only
            one_sample_t_test_design.inputs.global_calc_omit = True
            one_sample_t_test_design.inputs.no_grand_mean_scaling = True
            one_sample_t_test_design.inputs.global_normalization = 1 #None
            group_level_analysis.connect(
                one_sample_t_test_design, 'spm_mat_file', estimate_model, 'spm_mat_file')
            group_level_analysis.connect(
                get_contrasts, ('out_list', clean_list), one_sample_t_test_design, 'in_files')

        if method == 'equalRange':
            group_level_analysis.connect(
                get_equal_range_subjects, ('out_list', complete_subject_ids),
                get_contrasts, 'elements'
                )

        elif method == 'equalIndifference':
            group_level_analysis.connect(
                get_equal_indifference_subjects, ('out_list', complete_subject_ids),
                get_contrasts, 'elements'
                )

        elif method == 'groupComp':
            estimate_contrast.inputs.contrasts = [
                ('Eq range vs Eq indiff in loss', 'T', ['Group_{1}', 'Group_{2}'], [-1, 1])
                ]

            threshold.inputs.contrast_index = [1]
            threshold.synchronize = True

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

            # Node for the design matrix
            two_sample_t_test_design = Node(TwoSampleTTestDesign(),
                name = 'two_sample_t_test_design')
            two_sample_t_test_design.inputs.unequal_variance = True
            # no "overall grand mean scaling"
            two_sample_t_test_design.inputs.no_grand_mean_scaling = True
            two_sample_t_test_design.inputs.global_normalization = 1 # ANCOVA=no, no normalisation
            two_sample_t_test_design.inputs.use_implicit_threshold = True # implicit masking only
            two_sample_t_test_design.inputs.global_calc_omit = True # no "global calculation"
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

        # Datasink - save important files
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir
        group_level_analysis.connect(estimate_model, 'mask_image',
            data_sink, f'group_level_analysis_{method}_nsub_{nb_subjects}.@mask')
        group_level_analysis.connect(estimate_contrast, 'spm_mat_file',
            data_sink, f'group_level_analysis_{method}_nsub_{nb_subjects}.@spm_mat')
        group_level_analysis.connect(estimate_contrast, 'spmT_images',
            data_sink, f'group_level_analysis_{method}_nsub_{nb_subjects}.@T')
        group_level_analysis.connect(estimate_contrast, 'con_images',
            data_sink, f'group_level_analysis_{method}_nsub_{nb_subjects}.@con')
        group_level_analysis.connect(threshold, 'thresholded_map',
            data_sink, f'group_level_analysis_{method}_nsub_{nb_subjects}.@thresh')

        return group_level_analysis

    def get_group_level_outputs(self):
        """ Return all names for the files the group level analysis is supposed to generate. """

        # Handle equalRange and equalIndifference
        parameters = {
            'model': self.subject_level_models,
            'method': ['equalRange', 'equalIndifference'],
            'file': [
                'con_0001.nii', 'con_0002.nii', 'mask.nii', 'SPM.mat',
                'spmT_0001.nii', 'spmT_0002.nii',
                join('_threshold0', 'spmT_0001_thr.nii'), join('_threshold1', 'spmT_0002_thr.nii')
                ],
            'nb_subjects' : [str(len(self.subject_list))]
        }
        parameter_sets = product(*parameters.values())
        template = join(self.directories.output_dir,
            'group_level_analysis_{method}_nsub_{nb_subjects}', '_model_{model}',
            '{file}')

        return_list = [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

        # Handle groupComp
        parameters = {
            'model': self.subject_level_models,
            'file': [
                'con_0001.nii', 'mask.nii', 'SPM.mat', 'spmT_0001.nii',
                join('_threshold0', 'spmT_0001_thr.nii')
                ],
            'nb_subjects' : [str(len(self.subject_list))]
        }
        parameter_sets = product(*parameters.values())
        template = join(
            self.directories.output_dir,
            'group_level_analysis_groupComp_nsub_{nb_subjects}',
            '_model_{model}',
            '{file}'
            )

        return_list += [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

        return return_list

    def get_hypotheses_outputs(self):
        """ Return all hypotheses output file names. """
        nb_sub = len(self.subject_list)
        files = [
            # Hypothesis 1 - Positive parametric effect of gains
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_model_lossfirst', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_model_lossfirst', 'spmT_0001.nii'),
            # Hypothesis 2 - Positive parametric effect of gains
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_model_lossfirst', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_model_lossfirst', 'spmT_0001.nii'),
            # Hypothesis 3 - Positive parametric effect of gains
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_model_lossfirst', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_model_lossfirst', 'spmT_0001.nii'),
            # Hypothesis 4 - Positive parametric effect of gains
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_model_lossfirst', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_model_lossfirst', 'spmT_0001.nii'),
            # Hypothesis 5 - Negative parametric effect of losses
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_model_gainfirst', '_threshold1', 'spmT_0002_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_model_gainfirst', 'spmT_0002.nii'),
            # Hypothesis 6 - Negative parametric effect of losses
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_model_gainfirst', '_threshold1', 'spmT_0002_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_model_gainfirst', 'spmT_0002.nii'),
            # Hypothesis 7 - Positive parametric effect of losses
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_model_gainfirst', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_model_gainfirst', 'spmT_0001.nii'),
            # Hypothesis 8 - Positive parametric effect of losses
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_model_gainfirst', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_model_gainfirst', 'spmT_0001.nii'),
            # Hypothesis 9
            join(f'group_level_analysis_groupComp_nsub_{nb_sub}',
                '_model_gainfirst', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_groupComp_nsub_{nb_sub}',
                '_model_gainfirst', 'spmT_0001.nii')
        ]
        return [join(self.directories.output_dir, f) for f in files]
