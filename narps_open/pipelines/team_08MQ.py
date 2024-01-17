#!/usr/bin/python
# coding: utf-8

""" Write the work of NARPS team 08MQ using Nipype """

from os.path import join
from itertools import product

from nipype import Node, Workflow, MapNode
from nipype.interfaces.utility import IdentityInterface, Function, Merge, Split, Select
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.fsl import (
    # General usage
    FSLCommand, ImageStats,
    # Preprocessing
    FAST, BET, ErodeImage, PrepareFieldmap, MCFLIRT, SliceTimer,
    Threshold, Info, SUSAN, FLIRT, ApplyXFM, ConvertXFM,
    # Analyses
    Level1Design, FEATModel, L2Model, FILMGLS,
    FLAMEO, Randomise, MultipleRegressDesign
    )
from nipype.interfaces.fsl.utils import Merge as MergeImages
from nipype.interfaces.fsl.maths import MultiImageMaths
from nipype.algorithms.confounds import CompCor
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces.ants import Registration, WarpTimeSeriesImageMultiTransform

from narps_open.pipelines import Pipeline
from narps_open.data.task import TaskInformation
from narps_open.data.participants import get_group
from narps_open.core.common import (
    remove_file, list_intersection, elements_in_string, clean_list, list_to_file
    )

# Setup FSL
FSLCommand.set_default_output_type('NIFTI_GZ')

class PipelineTeam08MQ(Pipeline):
    """ A class that defines the pipeline of team 08MQ """

    def __init__(self):
        super().__init__()
        self.fwhm = 6.0
        self.team_id = '08MQ'
        self.contrast_list = ['1', '2', '3']
        self.run_level_contasts = [
            ('positive_effect_gain', 'T', ['gain', 'loss'], [1, 0]),
            ('positive_effect_loss', 'T', ['gain', 'loss'], [0, 1]),
            ('negative_effect_loss', 'T', ['gain', 'loss'], [0, -1])
        ]

    def get_preprocessing(self):
        """ Return a Nipype workflow describing the preprocessing part of the pipeline """

        # IdentityInterface node - allows to iterate over subjects and runs
        information_source = Node(IdentityInterface(
            fields = ['subject_id', 'run_id']),
            name = 'information_source')
        information_source.iterables = [
            ('run_id', self.run_list),
            ('subject_id', self.subject_list),
        ]

        # SelectFiles node - to select necessary files
        file_templates = {
            'anat': join('sub-{subject_id}', 'anat', 'sub-{subject_id}_T1w.nii.gz'),
            'func': join(
                'sub-{subject_id}', 'func', 'sub-{subject_id}_task-MGT_run-{run_id}_bold.nii.gz'
                ),
            'sbref': join(
                'sub-{subject_id}', 'func', 'sub-{subject_id}_task-MGT_run-{run_id}_sbref.nii.gz'
                ),
            'magnitude': join('sub-{subject_id}', 'fmap', 'sub-{subject_id}_magnitude1.nii.gz'),
            'phasediff': join('sub-{subject_id}', 'fmap', 'sub-{subject_id}_phasediff.nii.gz')
        }
        select_files = Node(SelectFiles(file_templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir

        # DataSink Node - store the wanted results in the wanted directory
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir

        # FAST Node - Bias field correction on anatomical images
        bias_field_correction = Node(FAST(), name = 'bias_field_correction')
        bias_field_correction.inputs.img_type = 1 # T1 image
        bias_field_correction.inputs.output_biascorrected = True

        # BET Node - Brain extraction for anatomical images
        brain_extraction_anat = Node(BET(), name = 'brain_extraction_anat')
        brain_extraction_anat.inputs.frac = 0.5

        # FAST Node - Segmentation of anatomical images
        segmentation_anat = Node(FAST(), name = 'segmentation_anat')
        segmentation_anat.inputs.no_bias = True # Bias field was already removed
        segmentation_anat.inputs.segments = False # Only output partial volume estimation
        segmentation_anat.inputs.probability_maps = False # Only output partial volume estimation

        # Split Node - Split probability maps as they output from the segmentation node
        #   outputs.out1 is CSF
        #   outputs.out2 is grey matter
        #   outputs.out3 is white matter
        split_segmentation_maps = Node(Split(), name = 'split_segmentation_maps')
        split_segmentation_maps.inputs.splits = [1, 1, 1]
        split_segmentation_maps.inputs.squeeze = True # Unfold one-element splits removing the list

        # ANTs Node - Normalization of anatomical images to T1 MNI152 space
        #   https://github.com/ANTsX/ANTs/wiki/Anatomy-of-an-antsRegistration-call
        normalization_anat = Node(Registration(), name = 'normalization_anat')
        normalization_anat.inputs.fixed_image = Info.standard_image('MNI152_T1_2mm_brain.nii.gz')
        normalization_anat.inputs.collapse_output_transforms = True
        normalization_anat.inputs.convergence_threshold = [1e-06]
        normalization_anat.inputs.convergence_window_size = [10]
        normalization_anat.inputs.dimension = 3
        normalization_anat.inputs.initial_moving_transform_com = True
        normalization_anat.inputs.radius_or_number_of_bins = [32, 32, 4]
        normalization_anat.inputs.sampling_percentage = [0.25, 0.25, 1]
        normalization_anat.inputs.sampling_strategy = ['Regular', 'Regular', 'None']
        normalization_anat.inputs.transforms = ['Rigid', 'Affine', 'SyN']
        normalization_anat.inputs.metric = ['MI', 'MI', 'CC']
        normalization_anat.inputs.transform_parameters = [(0.1,), (0.1,), (0.1, 3.0, 0.0)]
        normalization_anat.inputs.metric_weight = [1.0]*3
        normalization_anat.inputs.shrink_factors = [[8, 4, 2, 1]]*3
        normalization_anat.inputs.smoothing_sigmas = [[3, 2, 1, 0]]*3
        normalization_anat.inputs.sigma_units = ['vox']*3
        normalization_anat.inputs.number_of_iterations = [
            [1000, 500, 250, 100],
            [1000, 500, 250, 100],
            [100, 70, 50, 20]
            ]
        normalization_anat.inputs.use_histogram_matching = True
        normalization_anat.inputs.winsorize_lower_quantile = 0.005
        normalization_anat.inputs.winsorize_upper_quantile = 0.995

        # Threshold Node - create white-matter mask
        threshold_white_matter = Node(Threshold(), name = 'threshold_white_matter')
        threshold_white_matter.inputs.thresh = 1

        # Threshold Node - create CSF mask
        threshold_csf = Node(Threshold(), name = 'threshold_csf')
        threshold_csf.inputs.thresh = 1

        # ErodeImage Node - Erode white-matter mask
        erode_white_matter = Node(ErodeImage(), name = 'erode_white_matter')
        erode_white_matter.inputs.kernel_shape = 'sphere'
        erode_white_matter.inputs.kernel_size = 2.0 #mm

        # ErodeImage Node - Erode CSF mask
        erode_csf = Node(ErodeImage(), name = 'erode_csf')
        erode_csf.inputs.kernel_shape = 'sphere'
        erode_csf.inputs.kernel_size = 1.5 #mm

        # BET Node - Brain extraction of magnitude images
        brain_extraction_magnitude = Node(BET(), name = 'brain_extraction_magnitude')
        brain_extraction_magnitude.inputs.frac = 0.5

        # PrepareFieldmap Node - Convert phase and magnitude to fieldmap images
        convert_to_fieldmap = Node(PrepareFieldmap(), name = 'convert_to_fieldmap')

        # BET Node - Brain extraction for high contrast functional images
        brain_extraction_sbref = Node(BET(), name = 'brain_extraction_sbref')
        brain_extraction_sbref.inputs.frac = 0.3
        brain_extraction_sbref.inputs.mask = True
        brain_extraction_sbref.inputs.functional = False # 3D data

        # FLIRT Node - Align high contrast functional images to anatomical
        #   (i.e.: single-band reference images a.k.a. sbref)
        coregistration_sbref = Node(FLIRT(), name = 'coregistration_sbref')
        coregistration_sbref.inputs.interp = 'trilinear'
        coregistration_sbref.inputs.cost = 'bbr' # boundary-based registration

        # ConvertXFM Node - Inverse coregistration transform, to get anat to func transform
        inverse_func_to_anat = Node(ConvertXFM(), name = 'inverse_func_to_anat')
        inverse_func_to_anat.inputs.invert_xfm = True

        # BET Node - Brain extraction for functional images
        brain_extraction_func = Node(BET(), name = 'brain_extraction_func')
        brain_extraction_func.inputs.frac = 0.3
        brain_extraction_func.inputs.mask = True
        brain_extraction_func.inputs.functional = True

        # MCFLIRT Node - Motion correction of functional images
        motion_correction = Node(MCFLIRT(), name = 'motion_correction')
        motion_correction.inputs.cost = 'normcorr'
        motion_correction.inputs.interpolation = 'spline' # should be 'trilinear'
        motion_correction.inputs.save_plots = True # Save transformation parameters

        # Function Nodes get_slice_timings - Create a file with acquisition timing for each slide
        slice_timings = Node(Function(
            function = list_to_file,
            input_names = ['input_list', 'file_name'],
            output_names = ['output_file']
            ), name = 'slice_timings')
        slice_timings.inputs.input_list = TaskInformation()['SliceTiming']
        slice_timings.inputs.file_name = 'slice_timings.tsv'

        # SliceTimer Node - Slice time correction
        slice_time_correction = Node(SliceTimer(), name = 'slice_time_correction')
        slice_time_correction.inputs.time_repetition = TaskInformation()['RepetitionTime']

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

        # ApplyXFM Node - Alignment of white matter to functional space
        alignment_white_matter = Node(ApplyXFM(), name = 'alignment_white_matter')
        alignment_white_matter.inputs.apply_xfm = True
        alignment_white_matter.inputs.no_resample = True

        # ApplyXFM Node - Alignment of CSF to functional space
        alignment_csf = Node(ApplyXFM(), name = 'alignment_csf')
        alignment_csf.inputs.apply_xfm = True
        alignment_csf.inputs.no_resample = True

        # FLIRT Node - Alignment of functional data to anatomical space
        #   To save disk space we force isotropic resampling with 2.0 mm voxel dimension
        #   instead of 1.0 mm as reference file would suggest.
        #   We have to use FLIRT instead of ApplyXFM because there is a bug with
        #   apply_isoxfm and the latter.
        alignment_func_to_anat = Node(FLIRT(), name = 'alignment_func_to_anat')
        alignment_func_to_anat.inputs.apply_isoxfm = 2.0
        alignment_func_to_anat.inputs.no_resample = True

        # ApplyTransforms Node - Alignment of functional brain mask to anatomical space
        alignment_func_mask_to_anat = Node(ApplyXFM(), name = 'alignment_func_mask_to_anat')
        alignment_func_mask_to_anat.inputs.apply_xfm = True
        alignment_func_mask_to_anat.inputs.no_resample = True

        # Select Node - Change the order of transforms coming from ANTs Registration
        reverse_transform_order = Node(Select(), name = 'reverse_transform_order')
        reverse_transform_order.inputs.index = [1, 0]

        # ApplyWarp Node - Alignment of functional data to MNI space
        alignment_func_to_mni = Node(WarpTimeSeriesImageMultiTransform(),
            name = 'alignment_func_to_mni')
        alignment_func_to_mni.inputs.reference_image = \
            Info.standard_image('MNI152_T1_2mm_brain.nii.gz')

        # ApplyWarp Node - Alignment of functional data to MNI space
        alignment_func_mask_to_mni = Node(WarpTimeSeriesImageMultiTransform(),
            name = 'alignment_func_mask_to_mni')
        alignment_func_mask_to_mni.inputs.reference_image = \
            Info.standard_image('MNI152_T1_2mm_brain.nii.gz')

        # Merge Node - Merge the two masks (WM and CSF) in one input for the next node
        merge_masks = Node(Merge(2), name = 'merge_masks')

        # CompCor Node - Compute anatomical confounds (regressors of no interest in the model)
        #   from the WM and CSF masks
        compute_confounds = Node(CompCor(), name = 'compute_confounds')
        compute_confounds.inputs.num_components = 4
        compute_confounds.inputs.merge_method = 'union'
        compute_confounds.inputs.repetition_time = TaskInformation()['RepetitionTime']

        # Merge Node - Merge file names to be removed after datasink node is performed
        merge_removable_files = Node(Merge(8), name = 'merge_removable_files')
        merge_removable_files.inputs.ravel_inputs = True

        # Function Nodes remove_files - Remove sizeable files once they aren't needed
        remove_after_datasink = MapNode(Function(
            function = remove_file,
            input_names = ['_', 'file_name'],
            output_names = []
            ), name = 'remove_after_datasink', iterfield = 'file_name')
        remove_func = MapNode(Function(
            function = remove_file,
            input_names = ['_', 'file_name'],
            output_names = []
            ), name = 'remove_func', iterfield = 'file_name')

        preprocessing = Workflow(base_dir = self.directories.working_dir, name = 'preprocessing')
        preprocessing.config['execution']['stop_on_first_crash'] = 'true'
        preprocessing.connect([
            # Inputs
            (information_source, select_files, [
                ('subject_id', 'subject_id'), ('run_id', 'run_id')
                ]),

            # Anatomical images
            (select_files, bias_field_correction, [('anat', 'in_files')]),
            (bias_field_correction, brain_extraction_anat, [('restored_image', 'in_file')]),
            (brain_extraction_anat, segmentation_anat, [('out_file', 'in_files')]),
            (brain_extraction_anat, normalization_anat, [('out_file', 'moving_image')]),
            (segmentation_anat, split_segmentation_maps, [('partial_volume_files', 'inlist')]),
            (split_segmentation_maps, threshold_white_matter, [('out3', 'in_file')]),
            (split_segmentation_maps, threshold_csf, [('out1', 'in_file')]),
            (threshold_white_matter, erode_white_matter, [('out_file', 'in_file')]),
            (threshold_csf, erode_csf, [('out_file', 'in_file')]),
            (erode_white_matter, alignment_white_matter, [('out_file', 'in_file')]),
            (inverse_func_to_anat, alignment_white_matter, [('out_file', 'in_matrix_file')]),
            (select_files, alignment_white_matter, [('sbref', 'reference')]),
            (erode_csf, alignment_csf, [('out_file', 'in_file')]),
            (inverse_func_to_anat, alignment_csf, [('out_file', 'in_matrix_file')]),
            (select_files, alignment_csf, [('sbref', 'reference')]),
            (alignment_csf, merge_masks, [('out_file', 'in1')]),
            (alignment_white_matter, merge_masks, [('out_file', 'in2')]),

            # Field maps
            (select_files, brain_extraction_magnitude, [('magnitude', 'in_file')]),
            (brain_extraction_magnitude, convert_to_fieldmap, [('out_file', 'in_magnitude')]),
            (select_files, convert_to_fieldmap, [('phasediff', 'in_phase')]),

            # High contrast functional volume
            (select_files, brain_extraction_sbref, [('sbref', 'in_file')]),
            (brain_extraction_sbref, coregistration_sbref, [('out_file', 'in_file')]),
            (brain_extraction_anat, coregistration_sbref, [('out_file', 'reference')]),
            (split_segmentation_maps, coregistration_sbref, [('out3', 'wm_seg')]),
            (convert_to_fieldmap, coregistration_sbref, [('out_fieldmap', 'fieldmap')]),
            (coregistration_sbref, inverse_func_to_anat, [('out_matrix_file', 'in_file')]),

            # Functional images
            (select_files, brain_extraction_func, [('func', 'in_file')]),
            (brain_extraction_func, motion_correction, [('out_file', 'in_file')]),
            (select_files, motion_correction, [('sbref', 'ref_file')]),
            (slice_timings, slice_time_correction, [('output_file', 'custom_timings')]),
            (motion_correction, slice_time_correction, [('out_file', 'in_file')]),
            (slice_time_correction, smoothing, [('slice_time_corrected_file', 'in_file')]),
            (slice_time_correction, compute_median, [('slice_time_corrected_file', 'in_file')]),
            (brain_extraction_func, compute_median, [('mask_file', 'mask_file')]),
            (compute_median, smoothing, [
                (('out_stat', compute_brightness_threshold), 'brightness_threshold')
                ]),
            (smoothing, alignment_func_to_anat, [('smoothed_file', 'in_file')]),
            (coregistration_sbref, alignment_func_to_anat, [
                ('out_matrix_file', 'in_matrix_file')
                ]),
            (brain_extraction_anat, alignment_func_to_anat, [('out_file', 'reference')]),
            (brain_extraction_func, alignment_func_mask_to_anat, [('mask_file', 'in_file')]),
            (coregistration_sbref, alignment_func_mask_to_anat, [
                ('out_matrix_file', 'in_matrix_file')
                ]),
            (brain_extraction_anat, alignment_func_mask_to_anat, [('out_file', 'reference')]),
            (alignment_func_to_anat, alignment_func_to_mni, [('out_file', 'input_image')]),
            (alignment_func_mask_to_anat, alignment_func_mask_to_mni, [
                ('out_file', 'input_image')
                ]),
            (normalization_anat, reverse_transform_order, [('forward_transforms', 'inlist')]),
            (reverse_transform_order, alignment_func_to_mni, [('out', 'transformation_series')]),
            (reverse_transform_order, alignment_func_mask_to_mni, [
                ('out', 'transformation_series')
                ]),
            (merge_masks, compute_confounds, [('out', 'mask_files')]), #Masks are in the func space
            (slice_time_correction, compute_confounds, [
                ('slice_time_corrected_file', 'realigned_file')
                ]),

            # Outputs of preprocessing
            (motion_correction, data_sink, [('par_file', 'preprocessing.@par_file')]),
            (compute_confounds, data_sink, [
                ('components_file', 'preprocessing.@components_file')]),
            (alignment_func_to_mni, data_sink, [('output_image', 'preprocessing.@output_image')]),
            (alignment_func_mask_to_mni, data_sink, [
                ('output_image', 'preprocessing.@output_mask')]),

            # File removals
            (alignment_func_to_anat, remove_func, [('out_file', 'file_name')]),
            (alignment_func_to_mni, remove_func, [('output_image', '_')]),

            (motion_correction, merge_removable_files, [('out_file', 'in1')]),
            (slice_time_correction, merge_removable_files, [('slice_time_corrected_file', 'in2')]),
            (smoothing, merge_removable_files, [('smoothed_file', 'in3')]),
            (alignment_func_to_mni, merge_removable_files, [('output_image', 'in4')]),
            (brain_extraction_func, merge_removable_files, [('out_file', 'in5')]),
            (brain_extraction_anat, merge_removable_files, [('out_file', 'in6')]),
            (bias_field_correction, merge_removable_files, [('restored_image', 'in7')]),
            (normalization_anat, merge_removable_files, [('forward_transforms', 'in8')]),
            (merge_removable_files, remove_after_datasink, [('out', 'file_name')]),
            (data_sink, remove_after_datasink, [('out_file', '_')])
        ])

        return preprocessing

    def get_preprocessing_outputs(self):
        """ Return a list of the files generated by the preprocessing """

        parameters = {
            'subject_id': self.subject_list,
            'run_id': self.run_list,
            'file': [
                'components_file.txt',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_brain_mcf.nii.gz.par',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_brain_mcf_st_smooth_flirt_wtsimt.nii.gz',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_brain_mask_flirt_wtsimt.nii.gz'
            ]
        }
        parameter_sets = product(*parameters.values())
        template = join(
            self.directories.output_dir,
            'preprocessing',
            '_run_id_{run_id}_subject_id_{subject_id}',
            '{file}'
            )

        return [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

    def get_subject_information(event_file):
        """
        Extract information from an event file, to setup the model. 4 regressors are extracted :
        - event: a regressor with 4 second ON duration
        - gain : a parametric modulation of events corresponding to gain magnitude. Mean centred.
        - loss : a parametric modulation of events corresponding to loss magnitude. Mean centred.
        - response : a regressor with 1 for accept and -1 for reject. Mean centred.

        Parameters :
        - event_file : str, event file corresponding to the run and the subject to analyze

        Returns :
        - subject_info : list of Bunch containing event information
        """
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

        # Function Node get_subject_information - Get subject information from event files
        subject_information = Node(Function(
            function = self.get_subject_information,
            input_names = ['event_file'],
            output_names = ['subject_info']
            ), name = 'subject_information')

        # SpecifyModel Node - Generates a model
        specify_model = Node(SpecifyModel(), name = 'specify_model')
        specify_model.inputs.high_pass_filter_cutoff = 90
        specify_model.inputs.input_units = 'secs'
        specify_model.inputs.time_repetition = TaskInformation()['RepetitionTime']
        specify_model.inputs.parameter_source = 'FSL' # Source of motion parameters.

        # Level1Design Node - Generate files for first level computation
        model_design = Node(Level1Design(), 'model_design')
        model_design.inputs.bases = {
            'dgamma':{'derivs' : True} # Canonical double gamma HRF plus temporal derivative
            }
        model_design.inputs.interscan_interval = TaskInformation()['RepetitionTime']
        model_design.inputs.model_serial_correlations = True
        model_design.inputs.contrasts = self.run_level_contasts

        # FEATModel Node - Generate first level model
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
                ('subject_id', 'subject_id'), ('run_id', 'run_id')
                ]),
            (select_files, subject_information, [('event', 'event_file')]),
            (subject_information, specify_model, [('subject_info', 'subject_info')]),
            (select_files, specify_model, [('motion', 'realignment_parameters')]),
            (select_files, specify_model, [('func', 'functional_runs')]),
            (specify_model, model_design, [('session_info', 'session_info')]),
            (model_design, model_generation, [
                ('ev_files', 'ev_files'),
                ('fsf_files', 'fsf_file')]),
            (select_files, model_estimate, [('func', 'in_file')]),
            (model_generation, model_estimate, [
                ('con_file', 'tcon_file'),
                ('design_file', 'design_file')]),
            (model_estimate, data_sink, [('results_dir', 'run_level_analysis.@results')]),
            (model_generation, data_sink, [
                ('design_file', 'run_level_analysis.@design_file'),
                ('design_image', 'run_level_analysis.@design_img')]),
            ])

        return run_level_analysis

    def get_run_level_outputs(self):
        """ Return a list of the files generated by the run level analysis """

        parameters = {
            'run_id' : self.run_list,
            'subject_id' : self.subject_list,
            'file' : [
                'run0.mat',
                'run0.png'
            ]
        }
        parameter_sets = product(*parameters.values())
        template = join(
            self.directories.output_dir,
            'run_level_analysis', '_run_id_{run_id}_subject_id_{subject_id}','{file}'
            )
        return_list = [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

        parameters = {
            'run_id' : self.run_list,
            'subject_id' : self.subject_list,
            'contrast_id' : self.contrast_list,
            'file' : [
                join('results', 'cope{contrast_id}.nii.gz'),
                join('results', 'tstat{contrast_id}.nii.gz'),
                join('results', 'varcope{contrast_id}.nii.gz'),
                join('results', 'zstat{contrast_id}.nii.gz'),
            ]
        }
        parameter_sets = product(*parameters.values())
        template = join(
            self.directories.output_dir,
            'run_level_analysis', '_run_id_{run_id}_subject_id_{subject_id}','{file}'
            )

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
