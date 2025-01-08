#!/usr/bin/python
# coding: utf-8

""" Write the work of NARPS team V55J using Nipype """
from os.path import join
from itertools import product

from nipype import Workflow, Node, MapNode, JoinNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.algorithms.misc import Gunzip

from nipype.interfaces.spm import (
    Coregister, Smooth, OneSampleTTestDesign, EstimateModel, EstimateContrast,
    Level1Design, TwoSampleTTestDesign, RealignUnwarp,
    Normalize12, NewSegment, FieldMap, Threshold)
from nipype.interfaces.fsl import ExtractROI
from nipype.algorithms.modelgen import SpecifySPMModel
from nipype.interfaces.spm.base import Info as SPMInfo

from narps_open.pipelines import Pipeline
from narps_open.data.task import TaskInformation
from narps_open.data.participants import get_group
from narps_open.core.common import (
    remove_parent_directory, list_intersection, elements_in_string, clean_list
    )
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
        extract_tenth_image = Node(ExtractROI(), name = 'extract_tenth_image')
        extract_tenth_image.inputs.t_min = 10
        extract_tenth_image.inputs.t_size = 1
        extract_tenth_image.inputs.output_type='NIFTI'
        preprocessing.connect(gunzip_func, 'out_file', extract_tenth_image, 'in_file')

        # FIELDMAP - Calculate VDM routine of the FieldMap tool in SPM12
        # to the EPI image and to write out the distortion-corrected EPI image.
        # We set the structural image for comparison with the distortion-corrected EPI image and
        # matched the first to the latter.
        # For all other parameters we used the default values. We did not use Jacobian modulation.
        fieldmap = Node(FieldMap(), name = 'fieldmap')
        fieldmap.inputs.blip_direction = -1
        fieldmap.inputs.echo_times= [4.92, 7.38]
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
        # TODO apply to files ... but reverse tansform ?
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
        # We saved a bias-corrected version of the image and both inverse and forward deformation field
        # images.
        spm_tissues_file = join(SPMInfo.getinfo()['path'], 'tpm', 'TPM.nii')
        segmentation = Node(NewSegment(), name = 'segmentation')
        segmentation.inputs.write_deformation_fields = [True, True]
        segmentation.inputs.tissues = [
            [(spm_tissues_file, 1), 1, (True, False), (True, False)],
            [(spm_tissues_file, 2), 1, (True, False), (True, False)],
            [(spm_tissues_file, 3), 2, (True, False), (True, False)],
            [(spm_tissues_file, 4), 3, (True, False), (True, False)],
            [(spm_tissues_file, 5), 4, (True, False), (True, False)],
            [(spm_tissues_file, 6), 2, (True, False), (True, False)]
        ]
        preprocessing.connect(coregistration, 'coregistered_source', segmentation,'channel_files')

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
        normalize_func.inputs.fwhm = 6
        normalize_func.inputs.implicit_masking = False
        preprocessing.connect(normalize_func, 'normalized_files', smoothing, 'in_files')

        # DATASINK - Store the wanted results in the wanted repository
        data_sink = Node(DataSink(), name='data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir
        preprocessing.connect(
            motion_correction, 'realignment_parameters', data_sink, 'preprocess.@parameters')
        preprocessing.connect(smoothing, 'smoothed_files', data_sink, 'preprocess.@smoothing')
        preprocessing.connect(
            segmentation, 'native_class_images', data_sink, 'preprocess.@seg_maps_native')
        preprocessing.connect(
            segmentation, 'normalized_class_images', data_sink, 'preprocess.@seg_maps_norm')

        # Remove large files, if requested
        if Configuration()['pipelines']['remove_unused_data']:

            # Merge Node - Merge func file names to be removed after datasink node is performed
            merge_removable_func_files = Node(Merge(...), name = 'merge_removable_func_files')
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
            merge_removable_anat_files = Node(Merge(...), name = 'merge_removable_anat_files')
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
            'swusub-{subject_id}_task-MGT_run-{run_id}_bold.nii')]

        # Motion parameters file
        templates += [join(
            self.directories.output_dir,
            'preprocessing', '_subject_id_{subject_id}', '_run_id_{run_id}',
            'rp_sub-{subject_id}_task-MGT_run-{run_id}_bold.txt')]

        # Segmentation class images (native)
        templates += [join(
            self.directories.output_dir,
            'preprocessing', '_subject_id_{subject_id}', '_run_id_{run_id}',
            'rp_sub-{subject_id}_task-MGT_run-{run_id}_bold_roi.txt')]

        # Segmentation class images (normalized)
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

    def get_subject_information(event_file: str, short_run_id: int):
        """
        Create a Bunch of subject event information for specifySPMModel.

        Parameters :
        - event_file: str, events file for a run of a subject
        - short_run_id: str, an identifier for the run corresponding to the event_file
            must be '1' for the first run, '2' for the second run, etc.

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
                    param = [weights_gain[gain], weights_loss[loss]]
                ),
                None,
                None
                ],
            regressor_names = None,
            regressors = None
        )

    def compute_mask(wc1_file, wc2_file, wc3_file):
        """
        Function to compute mask to which.

        Parameters:
            - wc1_file: str, file wc1 obtained using segmentation function from SPM
            - wc2_file: str, file wc2 obtained using segmentation function from SPM
            - wc3_file: str, file wc3 obtained using segmentation function from SPM
        """
        from os import makdir
        from os.path import join, isdir

        from nilearn import masking, plotting
        from nibabel import Nifti1Image, save

        masks_list = []
        for mask in [wc1_file, wc2_file, wc3_file]:
            mask_img = nib.load(mask)
            mask_data = mask_img.get_fdata()
            mask_affine = mask_img.affine
            masks_list.append(mask_data)

        mask_img = masks_list[0] + masks_list[1] + masks_list[2] > 0.3
        mask_img = mask_img.astype('float64')
        mask = Nifti1Image(mask_img, mask_affine)
        if not os.path.isdir(join(result_dir, output_dir, 'subject_level_analysis')):
            os.mkdir(join(result_dir, output_dir, 'subject_level_analysis'))
        if not os.path.isdir(join(result_dir, output_dir, 'subject_level_analysis', f"_subject_id_{subject_id}")):
            os.mkdir(join(result_dir, output_dir, 'subject_level_analysis', f"_subject_id_{subject_id}"))
        mask_path = join(result_dir, output_dir, 'subject_level_analysis', f"_subject_id_{subject_id}", 'computed_mask.nii')
        save(mask, mask_path)

        return mask_path

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
            'wc1': join(self.directories.output_dir, 'preprocessing',
                '_subject_id_{subject_id}', 'wc1sub-{subject_id}_T1w.nii'),
            'wc2': join(self.directories.output_dir, 'preprocessing',
                '_subject_id_{subject_id}', 'wc2sub-{subject_id}_T1w.nii'),
            'wc3': join(self.directories.output_dir, 'preprocessing',
                '_subject_id_{subject_id}', 'wc3sub-{subject_id}_T1w.nii')
        }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir
        subject_level_analysis.connect(information_source, 'subject_id', select_files, 'subject_id')

        # FUNCTION node get_subject_information - get subject specific condition information
        subject_information = MapNode(Function(
                function = self.get_subject_information,
                input_names = ['event_file', 'short_run_id'],
                output_names = ['subject_info']),
            name = 'subject_information', iterfield = ['event_file', 'short_run_id'])
        subject_information.inputs.short_run_id = list(range(1, len(self.run_list) + 1))
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
        # subject_level_analysis.connect(
        #    select_files, 'parameters', specify_model, 'realignment_parameters')

        # FUNCTION node compute_mask - Compute mask from wc1, wc2, wc3 files
        compute_mask = Node(Function(
            function = self.compute_mask),
            input_names = ['wc1_file', 'wc2_file', 'wc3_file'],
            output_names = ['mask'],
            name = 'compute_mask')
        subject_level_analysis.connect(selectfiles, 'wc1', compute_mask, 'wc1_file')
        subject_level_analysis.connect(selectfiles, 'wc2', compute_mask, 'wc2_file')
        subject_level_analysis.connect(selectfiles, 'wc3', compute_mask, 'wc3_file')

        # LEVEL1 DESIGN - generates an SPM design matrix
        model_design = Node(Level1Design(), name='model_design')
        model_design.inputs.bases = {'hrf': {'derivs': [0, 0]}}
        model_design.inputs.timing_units = 'secs'
        model_design.inputs.interscan_interval = TaskInformation()['RepetitionTime']
        model_design.inputs.model_serial_correlations = 'AR(1)'
        subject_level_analysis.connect(compute_mask, 'mask', model_design, 'mask_image')
        subject_level_analysis.connect(specify_model, 'session_info', model_design, 'session_info')

        # ESTIMATE MODEL - estimate the parameters of the model
        model_estimate = Node(EstimateModel(), name = 'model_estimate')
        model_estimate.inputs.estimation_method = {'Classical': 1}
        (model_design, model_estimate, [('spm_mat_file', 'spm_mat_file')]),

        # ESTIMATE CONTRAST - estimates contrasts
        contrast_estimate = Node(EstimateContrast(), name="contrast_estimate")
        contrast_estimate.inputs.contrasts = self.subject_level_contrasts
        subject_level_analysis.connect(
            model_estimate, 'spm_mat_file', contrast_estimate, 'spm_mat_file')
        subject_level_analysis.connect(
            model_estimate, 'beta_images', contrast_estimate, 'beta_images')
        subject_level_analysis.connect(
            model_estimate, 'residual_image', contrast_estimate, 'residual_image')

        # DATASINK - store the wanted results in the wanted repository
        datasink = Node(DataSink(base_directory=result_dir, container=output_dir), name='datasink')

        """
                            (model_estimate, datasink, [('mask_image', 'subject_level_analysis.@mask')]),
                            (contrast_estimate, datasink, [('con_images', 'subject_level_analysis.@con_images'),
                                                                    ('spmT_images', 'subject_level_analysis.@spmT_images'),
                                                                    ('spm_mat_file', 'subject_level_analysis.@spm_mat_file')])
        """
        return subject_level_analysis


    def get_subject_level_outputs(self):
        """ Return the names of the files the subject level analysis is supposed to generate. """

        # Contrat maps
        templates = [join(self.directories.output_dir, f'subject_level_analysis',
            '_subject_id_{subject_id}', f'con_{c}.nii')\
            for c in self.contrast_list]

        # SPM.mat file
        templates += [join(self.directories.output_dir, f'subject_level_analysis',
            '_subject_id_{subject_id}', 'SPM.mat')]

        # spmT maps
        templates += [join(self.directories.output_dir, f'subject_level_analysis',
            '_subject_id_{subject_id}', f'spmT_{contrast_id}.nii')\
            for contrast_id in self.contrast_list]

        # Format with subject_ids
        return_list = []
        for template in templates:
            return_list += [template.format(subject_id = s) for s in self.subject_list]

        return return_list


    def get_subset_contrasts(file_list, method, subject_list, participants_file):
        """
        Parameters :
        - file_list : original file list selected by selectfiles node
        - subject_list : list of subject IDs that are in the wanted group for the analysis
        - participants_file: str, file containing participants characteristics
        - method: str, one of "equalRange", "equalIndifference" or "groupComp"

        This function return the file list containing only the files belonging to subject in the wanted group.
        """
        equalIndifference_id = []
        equalRange_id = []
        equalIndifference_files = []
        equalRange_files = []

        with open(participants_file, 'rt') as f:
                next(f)  # skip the header

                for line in f:
                    info = line.strip().split()

                    if info[0][-3:] in subject_list and info[1] == "equalIndifference":
                        equalIndifference_id.append(info[0][-3:])
                    elif info[0][-3:] in subject_list and info[1] == "equalRange":
                        equalRange_id.append(info[0][-3:])

        for file in file_list:
            sub_id = file.split('/')
            if sub_id[-2][-3:] in equalIndifference_id:
                equalIndifference_files.append(file)
            elif sub_id[-2][-3:] in equalRange_id:
                equalRange_files.append(file)

        return equalIndifference_id, equalRange_id, equalIndifference_files, equalRange_files


    def get_l2_analysis(subject_list, n_sub, contrast_list, method, exp_dir, result_dir, working_dir, output_dir):
        """
        Returns the 2nd level of analysis workflow.

        Parameters:
            - exp_dir: str, directory where raw data are stored
            - result_dir: str, directory where results will be stored
            - working_dir: str, name of the sub-directory for intermediate results
            - output_dir: str, name of the sub-directory for final results
            - subject_list: list of str, list of subject for which you want to do the preprocessing
            - contrast_list: list of str, list of contrasts to analyze
            - n_sub: float, number of subjects used to do the analysis
            - method: one of "equalRange", "equalIndifference" or "groupComp"

        Returns:
            - l2_analysis: Nipype WorkFlow
        """
        # information_source - a function free node to iterate over the list of subject names
        information_source_groupanalysis = Node(IdentityInterface(fields=['contrast_id', 'subjects'],
                                                          subjects = subject_list),
                          name="information_source_groupanalysis")

        information_source_groupanalysis.iterables = [('contrast_id', contrast_list)]

        # SelectFiles
        contrast_file = join(self.directories.output_dir, 'subject_level_analysis', '_subject_id_*', "con_00{contrast_id}.nii")

        participants_file = join(exp_dir, 'participants.tsv')

        templates = {'contrast' : contrast_file, 'participants' : participants_file}

        selectfiles_groupanalysis = Node(SelectFiles(templates, base_directory=result_dir, force_list= True),
                           name="selectfiles_groupanalysis")

        # Datasink node : to save important files
        datasink_groupanalysis = Node(DataSink(base_directory = result_dir, container = output_dir),
                                      name = 'datasink_groupanalysis')

        # Node to select subset of contrasts
        sub_contrasts = Node(Function(input_names = ['file_list', 'method', 'subject_list', 'participants_file'],
                                     output_names = ['equalIndifference_id', 'equalRange_id', 'equalIndifference_files', 'equalRange_files'],
                                     function = get_subset_contrasts),
                            name = 'sub_contrasts')

        sub_contrasts.inputs.method = method

        ## Estimate model
        estimate_model = Node(EstimateModel(estimation_method={'Classical':1}), name = "estimate_model")

        ## Estimate contrasts
        estimate_contrast = Node(EstimateContrast(group_contrast=True),
                                 name = "estimate_contrast")

        ## Create thresholded maps
        threshold = MapNode(Threshold(use_fwe_correction = False, height_threshold = 0.001), name = "threshold", iterfield = ["stat_image", "contrast_index"])

        l2_analysis = Workflow(base_dir = join(result_dir, working_dir), name = f"l2_analysis_{method}_nsub_{n_sub}")

        l2_analysis.connect([(information_source_groupanalysis, selectfiles_groupanalysis, [('contrast_id', 'contrast_id')]),
            (information_source_groupanalysis, sub_contrasts, [('subjects', 'subject_list')]),
            (selectfiles_groupanalysis, sub_contrasts, [('contrast', 'file_list'), ('participants', 'participants_file')]),
            (estimate_model, estimate_contrast, [('spm_mat_file', 'spm_mat_file'),
                ('residual_image', 'residual_image'),
                ('beta_images', 'beta_images')]),
            (estimate_contrast, threshold, [('spm_mat_file', 'spm_mat_file'),
                ('spmT_images', 'stat_image')]),
            (estimate_model, datasink_groupanalysis, [('mask_image', f"l2_analysis_{method}_nsub_{n_sub}.@mask")]),
            (estimate_contrast, datasink_groupanalysis, [('spm_mat_file', f"l2_analysis_{method}_nsub_{n_sub}.@spm_mat"),
                ('spmT_images', f"l2_analysis_{method}_nsub_{n_sub}.@T"),
                ('con_images', f"l2_analysis_{method}_nsub_{n_sub}.@con")]),
            (threshold, datasink_groupanalysis, [('thresholded_map', f"l2_analysis_{method}_nsub_{n_sub}.@thresh")])])

        if method=='equalRange' or method=='equalIndifference':
            contrasts = [('Group', 'T', ['mean'], [1]), ('Group', 'T', ['mean'], [-1])]
            ## Specify design matrix
            one_sample_t_test_design = Node(OneSampleTTestDesign(), name = "one_sample_t_test_design")

            l2_analysis.connect([(sub_contrasts, one_sample_t_test_design, [(f"{method}_files", 'in_files')]),
                (one_sample_t_test_design, estimate_model, [('spm_mat_file', 'spm_mat_file')])])

            threshold.inputs.contrast_index = [1, 2]
            threshold.synchronize = True

        elif method == 'groupComp':
            contrasts = [('Eq range vs Eq indiff in loss', 'T', ['Group_{1}', 'Group_{2}'], [1, -1])]
            # Node for the design matrix
            two_sample_t_test_design = Node(TwoSampleTTestDesign(), name = 'two_sample_t_test_design')

            l2_analysis.connect([(sub_contrasts, two_sample_t_test_design, [('equalRange_files', "group1_files"),
                ('equalIndifference_files', 'group2_files')]),
                (two_sample_t_test_design, estimate_model, [("spm_mat_file", "spm_mat_file")])])

            threshold.inputs.contrast_index = [1]
            threshold.synchronize = True

        estimate_contrast.inputs.contrasts = contrasts

        return l2_analysis


    def reorganize_results(result_dir, output_dir, n_sub, team_ID):
        """
        Reorganize the results to analyze them.

        Parameters:
            - result_dir: str, directory where results will be stored
            - output_dir: str, name of the sub-directory for final results
            - n_sub: float, number of subject used for the analysis
            - team_ID: str, ID of the team to reorganize results

        """
        from os.path import join as join
        import os
        import shutil
        import gzip

        h1 = join(result_dir, output_dir, f"l2_analysis_equalIndifference_nsub_{n_sub}", '_contrast_id_01')
        h2 = join(result_dir, output_dir, f"l2_analysis_equalRange_nsub_{n_sub}", '_contrast_id_01')
        h3 = join(result_dir, output_dir, f"l2_analysis_equalIndifference_nsub_{n_sub}", '_contrast_id_01')
        h4 = join(result_dir, output_dir, f"l2_analysis_equalRange_nsub_{n_sub}", '_contrast_id_01')
        h5 = join(result_dir, output_dir, f"l2_analysis_equalIndifference_nsub_{n_sub}", '_contrast_id_02')
        h6 = join(result_dir, output_dir, f"l2_analysis_equalRange_nsub_{n_sub}", '_contrast_id_02')
        h7 = join(result_dir, output_dir, f"l2_analysis_equalIndifference_nsub_{n_sub}", '_contrast_id_02')
        h8 = join(result_dir, output_dir, f"l2_analysis_equalRange_nsub_{n_sub}", '_contrast_id_02')
        h9 = join(result_dir, output_dir, f"l2_analysis_groupComp_nsub_{n_sub}", '_contrast_id_02')

        h = [h1, h2, h3, h4, h5, h6, h7, h8, h9]

        repro_unthresh = [join(filename, "spmT_0002.nii") if i in [4, 5] else join(filename,
                         "spmT_0001.nii") for i, filename in enumerate(h)]

        repro_thresh = [join(filename, "_threshold1",
             "spmT_0002_thr.nii") if i in [4, 5] else join(filename,
              "_threshold0", "spmT_0001_thr.nii")  for i, filename in enumerate(h)]

        if not os.path.isdir(join(result_dir, "NARPS-reproduction")):
            os.mkdir(join(result_dir, "NARPS-reproduction"))

        for i, filename in enumerate(repro_unthresh):
            f_in = filename
            f_out = join(result_dir, "NARPS-reproduction", f"team_{team_ID}_nsub_{n_sub}_hypo{i+1}_unthresholded.nii")
            shutil.copyfile(f_in, f_out)

        for i, filename in enumerate(repro_thresh):
            f_in = filename
            f_out = join(result_dir, "NARPS-reproduction", f"team_{team_ID}_nsub_{n_sub}_hypo{i+1}_thresholded.nii")
            shutil.copyfile(f_in, f_out)

        print(f"Results files of team {team_ID} reorganized.")


