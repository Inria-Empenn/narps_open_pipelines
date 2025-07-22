#!/usr/bin/python
# coding: utf-8

""" Write the work of NARPS team V55J using Nipype """
from os.path import join
from itertools import product

from nipype import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Function, Merge
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.algorithms.misc import Gunzip

from nipype.interfaces.spm import (
    Coregister, Smooth, OneSampleTTestDesign, EstimateModel, EstimateContrast,
    Level1Design, TwoSampleTTestDesign, RealignUnwarp,
    Normalize12, NewSegment, FieldMap, Threshold)
from nipype.algorithms.modelgen import SpecifySPMModel
from nipype.interfaces.spm.base import Info as SPMInfo

from narps_open.pipelines import Pipeline
from narps_open.data.task import TaskInformation
from narps_open.data.participants import get_group
from narps_open.core.common import (
    remove_parent_directory, list_intersection, elements_in_string, clean_list
    )
from narps_open.core.image import get_image_timepoints
from narps_open.utils.configuration import Configuration

class PipelineTeamV55J(Pipeline):
    """ A class that defines the pipeline of team V55J. """

    def __init__(self):
        super().__init__()
        self.fwhm = 6.0
        self.team_id = 'V55J'

        # Create contrasts
        conditions = ['trial', 'trialxgain^1', 'trialxloss^1']
        self.subject_level_contrasts = [
            ('effect_of_gain', 'T', conditions, [0, 1, 0]),
            ('effect_of_loss', 'T', conditions, [0, 0, 1])
        ]
        self.contrast_list = ['0001', '0002']

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
        file_templates = {
            'anat': join('sub-{subject_id}', 'anat', 'sub-{subject_id}_T1w.nii.gz'),
            'magnitude': join('sub-{subject_id}', 'fmap', 'sub-{subject_id}_magnitude1.nii.gz'),
            'phasediff': join('sub-{subject_id}', 'fmap', 'sub-{subject_id}_phasediff.nii.gz')
            }
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
        gunzip_magnitude = Node(Gunzip(), name = 'gunzip_magnitude')
        gunzip_phasediff = Node(Gunzip(), name = 'gunzip_phasediff')
        preprocessing.connect(select_subject_files, 'anat', gunzip_anat, 'in_file')
        preprocessing.connect(select_subject_files, 'magnitude', gunzip_magnitude, 'in_file')
        preprocessing.connect(select_subject_files, 'phasediff', gunzip_phasediff, 'in_file')
        preprocessing.connect(select_run_files, 'func', gunzip_func, 'in_file')

        # EXTRACTROI - get the image 10 in func file
        # "For each run, we selected image 10 to distortion correct and asked to match the VDM file"
        extract_tenth_image = Node(Function(
            function = get_image_timepoints,
            input_names = ['in_file', 'start_time_point', 'end_time_point'],
            output_names = ['roi_file']
            ), name = 'extract_tenth_image')
        extract_tenth_image.inputs.start_time_point = 9 # 0-based 10th image
        extract_tenth_image.inputs.end_time_point = 9
        preprocessing.connect(gunzip_func, 'out_file', extract_tenth_image, 'in_file')

        # FIELDMAP - Calculate VDM routine of the FieldMap tool in SPM12
        # to the EPI image and to write out the distortion-corrected EPI image.
        # We set the structural image for comparison with the distortion-corrected EPI image and
        # matched the first to the latter.
        # For all other parameters we used the default values. We did not use Jacobian modulation.
        fieldmap = Node(FieldMap(), name = 'fieldmap')
        fieldmap.inputs.blip_direction = -1
        fieldmap.inputs.echo_times = (4.92, 7.38)
        fieldmap.inputs.total_readout_time = 29.15
        fieldmap.inputs.matchanat = True
        fieldmap.inputs.matchvdm = True
        fieldmap.inputs.writeunwarped = True
        fieldmap.inputs.maskbrain = False
        fieldmap.inputs.thresh = 0
        preprocessing.connect(gunzip_anat, 'out_file', fieldmap, 'anat_file')
        preprocessing.connect(extract_tenth_image, 'roi_file', fieldmap, 'epi_file')
        preprocessing.connect(gunzip_magnitude, 'out_file', fieldmap, 'magnitude_file')
        preprocessing.connect(gunzip_phasediff, 'out_file', fieldmap, 'phase_file')

        # REALIGN UNWARP - Motion correction
        motion_correction = Node(RealignUnwarp(), name = 'motion_correction')
        motion_correction.inputs.interp = 7
        motion_correction.inputs.register_to_mean = True
        motion_correction.inputs.quality = 1
        motion_correction.inputs.reslice_mask = False
        preprocessing.connect(gunzip_func, 'out_file', motion_correction, 'in_files')
        preprocessing.connect(fieldmap, 'vdm', motion_correction, 'phase_map')

        # COREGISTER - Coregistration from anat to realigned func mean image
        # We kept the default values for all other parameters.
        # TODO apply to files ... but reverse transform ?
        coregistration = Node(Coregister(), name = 'coregistration')
        coregistration.inputs.jobtype = 'estimate'
        coregistration.inputs.write_mask = False
        preprocessing.connect(gunzip_anat, 'out_file', coregistration, 'source')
        preprocessing.connect(motion_correction, 'mean_image', coregistration, 'target')
        preprocessing.connect(
            motion_correction, 'realigned_unwarped_files', coregistration, 'apply_to_files')

        # NEWSEGMENT - Segmentation of anat
        # We performed segmentation on the structural image for each subject by using the "Segment"
        # routine in SPM12, with default values for each parameter and using the template tissue
        # probability maps (grey matter, white matter, CSF, bone, soft tissue, and air/background)
        # in the tpm folder of SPM12.
        # We saved a bias-corrected version of the image and both inverse
        # and forward deformation field images.
        spm_tissues_file = join(SPMInfo.getinfo()['path'], 'tpm', 'TPM.nii')
        segmentation = Node(NewSegment(), name = 'segmentation')
        segmentation.inputs.write_deformation_fields = [True, True]
        segmentation.inputs.tissues = [
            [(spm_tissues_file, 1), 1, (True, False), (True, False)], # TODO change gaussians ?
            [(spm_tissues_file, 2), 1, (True, False), (True, False)],
            [(spm_tissues_file, 3), 2, (True, False), (True, False)],
            [(spm_tissues_file, 4), 3, (True, False), (True, False)],
            [(spm_tissues_file, 5), 4, (True, False), (True, False)],
            [(spm_tissues_file, 6), 2, (True, False), (True, False)]
        ]
        preprocessing.connect(coregistration, 'coregistered_source', segmentation, 'channel_files')

        # NORMALIZE12 - Normalization of func
        # We used the "Normalise: Write" routine in SPM12. We set the motion-corrected EPI images
        # for each run as images to resample and the spatial normalization deformation field file
        # obtained with the "Segment" routine as deformation field for the normalization procedure.
        # We used default values for the bounding box and set voxel size to 2 x 2 x 2.4 mm and
        # interpolation method to 7th degree B-spline.
        normalize_func = Node(Normalize12(), name = 'normalize_func')
        normalize_func.inputs.jobtype = 'write'
        normalize_func.inputs.write_voxel_sizes = [2, 2, 2.4]
        normalize_func.inputs.write_interp = 7
        preprocessing.connect(
            coregistration, 'coregistered_files', normalize_func, 'apply_to_files')
        preprocessing.connect(
            segmentation, 'forward_deformation_field', normalize_func, 'deformation_file')

        # SMOOTH - Smoothing of func
        # We used the "Smooth" routine in SPM12. We selected the normalized EPI images and set
        # the FWHM of the Gaussian smoothing kernel to 6mm. We used the default values for
        # the other parameters.
        smoothing = Node(Smooth(), name = 'smoothing')
        smoothing.inputs.fwhm = 6
        smoothing.inputs.implicit_masking = False
        preprocessing.connect(normalize_func, 'normalized_files', smoothing, 'in_files')

        # DATASINK - Store the wanted results in the wanted repository
        data_sink = Node(DataSink(), name='data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir
        preprocessing.connect(smoothing, 'smoothed_files', data_sink, 'preprocessing.@smoothing')
        preprocessing.connect(
            segmentation, 'normalized_class_images', data_sink, 'preprocessing.@seg_maps_norm')

        # Remove large files, if requested
        if Configuration()['pipelines']['remove_unused_data']:

            # Merge Node - Merge func file names to be removed after datasink node is performed
            merge_removable_func_files = Node(Merge(7), name = 'merge_removable_func_files')
            merge_removable_func_files.inputs.ravel_inputs = True

            # Function Nodes remove_files - Remove sizeable files once they aren't needed
            remove_func_after_datasink = MapNode(Function(
                function = remove_parent_directory,
                input_names = ['_', 'file_name'],
                output_names = []
                ), name = 'remove_func_after_datasink', iterfield = 'file_name')
            preprocessing.connect(gunzip_func, 'out_file', merge_removable_func_files, 'in1')
            preprocessing.connect(
                extract_tenth_image, 'roi_file', merge_removable_func_files, 'in2')
            preprocessing.connect(fieldmap, 'vdm', merge_removable_func_files, 'in3')
            preprocessing.connect(
                motion_correction, 'mean_image', merge_removable_func_files, 'in4')
            preprocessing.connect(
                coregistration, 'coregistered_source', merge_removable_func_files, 'in5')
            preprocessing.connect(smoothing, 'smoothed_files',  merge_removable_func_files, 'in6')
            preprocessing.connect(
                segmentation, 'native_class_images', merge_removable_func_files, 'in7')
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
            preprocessing.connect(gunzip_phasediff, 'out_file', merge_removable_anat_files, 'in2')
            preprocessing.connect(gunzip_magnitude, 'out_file', merge_removable_anat_files, 'in3')
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
            'swusub-{subject_id}_task-MGT_run-{run_id}_bold.nii')]

        # Segmentation class images (normalized)
        templates += [join(
            self.directories.output_dir,
            'preprocessing', '_subject_id_{subject_id}', '_run_id_{run_id}',
            f'wc{c}' + 'sub-{subject_id}_T1w.nii') for c in range(1, 7)]

        # Format with subject_ids
        return_list = []
        for template in templates:
            return_list += [template.format(subject_id = s, run_id = r)\
                for r in self.run_list for s in self.subject_list]

        return return_list

    def get_run_level_analysis(self):
        """ Return a Nipype workflow describing the run level analysis part of the pipeline """
        return None

    def get_subject_information(event_file: str):
        """
        Create a Bunch of subject event information for specifySPMModel.

        Parameters :
        - event_file: str, events file for a run of a subject

        Returns :
        - a Bunch corresponding to the event file
        """
        from nipype.interfaces.base import Bunch

        onsets_trial = []
        durations_trial = []
        weights_gain = []
        weights_loss = []
        onsets_accepting = []
        durations_accepting = []
        onsets_rejecting = []
        durations_rejecting = []

        with open(event_file, 'rt') as file:
            next(file)  # skip the header

            for line in file:
                info = line.strip().split()

                """
                The model contained 6 regressors per run:
                - One predictor with onset at the start of the trial and duration of 4s.
                - Two parametric modulators (one for gains, one for losses)
                were added to the trial onset predictor.
                The two parametric modulators were orthogonalized w.r.t. the main predictor,
                but were not orthogonalized w.r.t. one another.
                """
                onsets_trial.append(float(info[0]))
                durations_trial.append(4.0)
                weights_gain.append(float(info[2]))
                weights_loss.append(float(info[3]))

                """
                - Two predictors modelling the decision output, one for accepting the gamble
                and one for rejecting it (merging strong and weak decisions).
                The onset was defined as the beginning of the trial + RT
                and the duration was set to 0 (stick function).
                - One constant term for each run was included (SPM12 default design).
                """
                if 'accept' in info[5]:
                    onsets_accepting.append(float(info[0]) + float(info[4]))
                    durations_accepting.append(0.0)
                elif 'reject' in info[5]:
                    onsets_rejecting.append(float(info[0]) + float(info[4]))
                    durations_rejecting.append(0.0)

        # Create bunch
        return Bunch(
            conditions = ['trial', 'accepting', 'rejecting'],
            onsets = [onsets_trial, onsets_accepting, onsets_rejecting],
            durations = [durations_trial, durations_accepting, durations_rejecting],
            amplitudes = None,
            tmod = None,
            pmod = [
                Bunch(
                    name = ['gain', 'loss'],
                    poly = [1, 1],
                    param = [weights_gain, weights_loss]
                ),
                None,
                None
                ],
            regressor_names = None,
            regressors = None
        )

    def union_mask(masks: list, threshold: float = 0.0):
        """
        Compute union between masks, then binarize the result using a threshold value.

        Parameters:
            - masks: list, a list of .nii masks
            - threshold: float, values under threshold will be set to 0.0, other to 1.0

        Returns:
            - binarized union mask
        """
        from os.path import abspath

        from nibabel import Nifti1Image, load as nib_load, save as nib_save

        # Open mask files
        for mask_id, mask in enumerate(masks):
            mask_image = nib_load(mask)
            mask_data = mask_image.get_fdata()
            mask_affine = mask_image.affine

            # Perform mask union
            if mask_id == 0:
                mask_union = mask_data
            else:
                mask_union += mask_data

        # Binarize mask
        mask_union = mask_union > threshold

        # Output file
        mask_union = mask_union.astype('float64')
        binarized_mask_union = abspath('mask.nii')
        nib_save(Nifti1Image(mask_union, mask_affine), binarized_mask_union)

        return binarized_mask_union

    def get_subject_level_analysis(self):
        """ Return a nipype.WorkFlow describing the subject level analysis part of the pipeline """

        # Workflow initialization
        subject_level_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = 'subject_level_analysis')

        # IdentityInterface - To iterate on subjects
        information_source = Node(IdentityInterface(fields = ['subject_id']),
            name = 'information_source')
        information_source.iterables = [('subject_id', self.subject_list)]

        # SELECTFILES - to select necessary files
        templates = {
            'func': join(self.directories.output_dir, 'preprocessing',
                '_subject_id_{subject_id}', '_run_id_*',
                'swusub-{subject_id}_task-MGT_run-*_bold.nii'
            ),
            'event': join(self.directories.dataset_dir, 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-*_events.tsv'
            ),
            'wc1': join(self.directories.output_dir, 'preprocessing', '_subject_id_{subject_id}',
                '_run_id_*', 'wc1sub-{subject_id}_T1w.nii'),
            'wc2': join(self.directories.output_dir, 'preprocessing', '_subject_id_{subject_id}',
                '_run_id_*', 'wc2sub-{subject_id}_T1w.nii'),
            'wc3': join(self.directories.output_dir, 'preprocessing', '_subject_id_{subject_id}',
                '_run_id_*', 'wc3sub-{subject_id}_T1w.nii')
        }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir
        subject_level_analysis.connect(information_source, 'subject_id', select_files, 'subject_id')

        # FUNCTION node get_subject_information - get subject specific condition information
        subject_information = MapNode(Function(
                function = self.get_subject_information,
                input_names = ['event_file'],
                output_names = ['subject_info']),
            name = 'subject_information', iterfield = ['event_file'])
        subject_level_analysis.connect(select_files, 'event', subject_information, 'event_file')

        # SPECIFY MODEL - generates SPM-specific Model
        specify_model = Node(SpecifySPMModel(), name = 'specify_model')
        specify_model.inputs.concatenate_runs = True
        specify_model.inputs.input_units = 'secs'
        specify_model.inputs.output_units = 'secs'
        specify_model.inputs.time_repetition = TaskInformation()['RepetitionTime']
        specify_model.inputs.high_pass_filter_cutoff = 128
        subject_level_analysis.connect(
            subject_information, 'subject_info', specify_model, 'subject_info')
        subject_level_analysis.connect(select_files, 'func', specify_model, 'functional_runs')

        # MERGE - merge mask files into a list
        merge_masks = Node(Merge(3), name = 'merge_masks')
        subject_level_analysis.connect(select_files, 'wc1', merge_masks, 'in1')
        subject_level_analysis.connect(select_files, 'wc2', merge_masks, 'in2')
        subject_level_analysis.connect(select_files, 'wc3', merge_masks, 'in3')

        # FUNCTION node compute_mask - Compute mask from wc1, wc2, wc3 files
        mask_union = Node(Function(
            function = self.union_mask,
            input_names = ['masks', 'threshold'],
            output_names = ['binarized_mask_union']),
            name = 'mask_union')
        mask_union.inputs.threshold = 0.3
        subject_level_analysis.connect(merge_masks, 'out', mask_union, 'masks')

        # LEVEL1 DESIGN - generates an SPM design matrix
        model_design = Node(Level1Design(), name = 'model_design')
        model_design.inputs.bases = {'hrf': {'derivs': [0, 0]}}
        model_design.inputs.timing_units = 'secs'
        model_design.inputs.interscan_interval = TaskInformation()['RepetitionTime']
        model_design.inputs.model_serial_correlations = 'AR(1)'
        subject_level_analysis.connect(
            mask_union, 'binarized_mask_union', model_design, 'mask_image')
        subject_level_analysis.connect(specify_model, 'session_info', model_design, 'session_info')

        # ESTIMATE MODEL - estimate the parameters of the model
        model_estimate = Node(EstimateModel(), name = 'model_estimate')
        model_estimate.inputs.estimation_method = {'Classical': 1}
        subject_level_analysis.connect(
            model_design, 'spm_mat_file', model_estimate, 'spm_mat_file')

        # ESTIMATE CONTRAST - estimates contrasts
        contrast_estimate = Node(EstimateContrast(), name = 'contrast_estimate')
        contrast_estimate.inputs.contrasts = self.subject_level_contrasts
        subject_level_analysis.connect(
            model_estimate, 'spm_mat_file', contrast_estimate, 'spm_mat_file')
        subject_level_analysis.connect(
            model_estimate, 'beta_images', contrast_estimate, 'beta_images')
        subject_level_analysis.connect(
            model_estimate, 'residual_image', contrast_estimate, 'residual_image')

        # DATASINK - store the wanted results in the wanted repository
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir
        subject_level_analysis.connect(
            contrast_estimate, 'con_images', data_sink, 'subject_level_analysis.@con_images')
        subject_level_analysis.connect(
            contrast_estimate, 'spmT_images', data_sink, 'subject_level_analysis.@spmT_images')
        subject_level_analysis.connect(
            contrast_estimate, 'spm_mat_file', data_sink, 'subject_level_analysis.@spm_mat_file')

        return subject_level_analysis

    def get_subject_level_outputs(self):
        """ Return the names of the files the subject level analysis is supposed to generate. """

        # Contrat maps
        templates = [join(self.directories.output_dir, 'subject_level_analysis',
            '_subject_id_{subject_id}', f'con_{c}.nii') for c in self.contrast_list]

        # SPM.mat file
        templates += [join(self.directories.output_dir, 'subject_level_analysis',
            '_subject_id_{subject_id}', 'SPM.mat')]

        # spmT maps
        templates += [join(self.directories.output_dir, 'subject_level_analysis',
            '_subject_id_{subject_id}', f'spmT_{c}.nii') for c in self.contrast_list]

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
                fields = ['contrast_id']),
                name = 'information_source')
        information_source.iterables = [('contrast_id', self.contrast_list)]

        # SELECT FILES - select contrasts for all subjects
        templates = {
            'contrasts' : join('subject_level_analysis', '_subject_id_*', 'con_{contrast_id}.nii')
            }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.output_dir
        select_files.inputs.force_list = True
        group_level_analysis.connect(
            information_source, 'contrast_id', select_files, 'contrast_id')

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
        estimate_contrast.inputs.group_contrast=True
        group_level_analysis.connect(
            estimate_model, 'spm_mat_file', estimate_contrast, 'spm_mat_file')
        group_level_analysis.connect(
            estimate_model, 'residual_image', estimate_contrast, 'residual_image')
        group_level_analysis.connect(
            estimate_model, 'beta_images', estimate_contrast, 'beta_images')

        ## Create thresholded maps
        threshold = MapNode(Threshold(),
            name = 'threshold', iterfield = ['stat_image', 'contrast_index'])
        threshold.inputs.use_fwe_correction = False
        threshold.inputs.height_threshold = 0.001
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
        template = join(self.directories.output_dir,
            'group_level_analysis_{method}_nsub_{nb_subjects}', '_contrast_id_{contrast_id}',
            '{file}')

        return_list = [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

        # Handle groupComp
        parameters = {
            'contrast_id': self.contrast_list,
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
            '_contrast_id_{contrast_id}', '{file}'
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
                '_contrast_id_0001', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001', 'spmT_0001.nii'),
            # Hypothesis 2 - Positive parametric effect of gains
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001', 'spmT_0001.nii'),
            # Hypothesis 3 - Positive parametric effect of gains
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001', 'spmT_0001.nii'),
            # Hypothesis 4 - Positive parametric effect of gains
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001', 'spmT_0001.nii'),
            # Hypothesis 5 - Negative parametric effect of losses
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold1', 'spmT_0002_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0002.nii'),
            # Hypothesis 6 - Negative parametric effect of losses
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold1', 'spmT_0002_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0002.nii'),
            # Hypothesis 7 - Positive parametric effect of losses
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0001.nii'),
            # Hypothesis 8 - Positive parametric effect of losses
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0001.nii'),
            # Hypothesis 9
            join(f'group_level_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0001.nii')
        ]
        return [join(self.directories.output_dir, f) for f in files]
