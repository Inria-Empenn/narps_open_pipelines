#!/usr/bin/python
# coding: utf-8

""" Write the work of NARPS team 98BT using Nipype """
from os.path import join
from itertools import product

from nipype import Workflow, Node, MapNode, JoinNode
from nipype.interfaces.utility import IdentityInterface, Function, Rename, Merge
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.algorithms.misc import Gunzip
from nipype.interfaces.spm import (
    Coregister, OneSampleTTestDesign,
    EstimateModel, EstimateContrast, Level1Design,
    TwoSampleTTestDesign, RealignUnwarp, NewSegment, SliceTiming,
    DARTELNorm2MNI, FieldMap, Threshold
    )
from nipype.interfaces.spm.base import Info as SPMInfo
from nipype.interfaces.fsl import ExtractROI
from nipype.algorithms.modelgen import SpecifySPMModel
from niflow.nipype1.workflows.fmri.spm import create_DARTEL_template

from narps_open.pipelines import Pipeline
from narps_open.data.task import TaskInformation
from narps_open.data.participants import get_group
from narps_open.core.common import (
    remove_parent_directory, list_intersection, elements_in_string, clean_list
    )
from narps_open.utils.configuration import Configuration

class PipelineTeam98BT(Pipeline):
    """ A class that defines the pipeline of team 98BT. """

    def __init__(self):
        super().__init__()
        self.fwhm = 8.0
        self.team_id = '98BT'
        self.contrast_list = ['0001', '0002', '0003', '0004']

        # Define contrasts
        gain_conditions = [f'gamble_run{r}xgain_run{r}^1' for r in range(1,len(self.run_list) + 1)]
        loss_conditions = [f'gamble_run{r}xloss_run{r}^1' for r in range(1,len(self.run_list) + 1)]
        self.subject_level_contrasts = [
            ('pos_gain', 'T', gain_conditions, [1, 1, 1, 1]),
            ('pos_loss', 'T', loss_conditions, [1, 1, 1, 1]),
            ('neg_gain', 'T', gain_conditions, [-1, -1, -1, -1]),
            ('neg_loss', 'T', loss_conditions, [-1, -1, -1, -1])
            ]

    def get_dartel_template_sub_workflow(self):
        """
        Create a dartel workflow, as first part of the preprocessing.

        DARTEL allows to create a study-specific template in 3D volume space.
        This study template can then be used for normalizating each subject’s
        scans to the MNI space.

        Returns:
            - dartel : nipype.WorkFlow
        """
        # Infosource Node - To iterate on subjects
        information_source = Node(IdentityInterface(
            fields = ['subject_id']),
            name = 'information_source')
        information_source.iterables = ('subject_id', self.subject_list)

        # SelectFiles Node - to select necessary files
        template = {
            'anat' : join('sub-{subject_id}', 'anat', 'sub-{subject_id}_T1w.nii.gz')
        }
        select_files = Node(SelectFiles(template), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir

        # Gunzip node - SPM do not use .nii.gz files
        gunzip_anat = Node(Gunzip(), name = 'gunzip_anat')

        def get_dartel_input(structural_files):
            print(structural_files)
            return structural_files

        dartel_input = JoinNode(Function(
            function = get_dartel_input,
            input_names = ['structural_files'],
            output_names = ['structural_files']),
            name = 'dartel_input',
            joinsource = 'information_source',
            joinfield = 'structural_files')

        rename_dartel = MapNode(Rename(format_string = 'subject_id_%(subject_id)s_struct'),
            iterfield = ['in_file', 'subject_id'],
            name = 'rename_dartel')
        rename_dartel.inputs.subject_id = self.subject_list
        rename_dartel.inputs.keep_ext = True

        dartel_sub_workflow = create_DARTEL_template(name = 'dartel_sub_workflow')
        dartel_sub_workflow.inputs.inputspec.template_prefix = 'template'

        # DataSink Node - store the wanted results in the wanted repository
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir

        # Create dartel workflow and connect its nodes
        dartel_workflow = Workflow(base_dir = self.directories.working_dir, name = 'dartel_workflow')
        dartel_workflow.connect([
            (information_source, select_files, [('subject_id', 'subject_id')]),
            (select_files, gunzip_anat, [('anat', 'in_file')]),
            (gunzip_anat, dartel_input, [('out_file', 'structural_files')]),
            (dartel_input, rename_dartel, [('structural_files', 'in_file')]),
            (rename_dartel, dartel_sub_workflow, [('out_file', 'inputspec.structural_files')]),
            (dartel_sub_workflow, data_sink, [
                ('outputspec.template_file', 'dartel_template.@template_file'),
                ('outputspec.flow_fields', 'dartel_template.@flow_fields')])
            ])

        # Remove large files, if requested
        if Configuration()['pipelines']['remove_unused_data']:

            # Function Nodes remove_parent_directory - Remove gunziped files
            remove_gunzip = Node(Function(
                function = remove_parent_directory,
                input_names = ['_', 'file_name'],
                output_names = []
                ), name = 'remove_gunzip')

            # Add connections
            dartel_workflow.connect([
                (gunzip_anat, remove_gunzip, [('out_file', 'file_name')]),
                (data_sink, remove_gunzip, [('out_file', '_')])
            ])

        return dartel_workflow

    def get_fieldmap_info(fieldmap_info_file, magnitude):
        """
        Function to get information necessary to compute the fieldmap.

        Parameters:
            - fieldmap_info_file: str, file with fieldmap information
            - magnitude: list of str, list of magnitude files

        Returns:
            - echo_times: tuple of floats, echo time obtained from fieldmap information file
            - magnitude_file: str, necessary file to compute fieldmap
        """
        from json import load

        with open(fieldmap_info_file, 'rt') as file:
            fieldmap_info = load(file)

        short_echo_time = min(float(fieldmap_info['EchoTime1']), float(fieldmap_info['EchoTime2']))
        long_echo_time = max(float(fieldmap_info['EchoTime1']), float(fieldmap_info['EchoTime2']))

        if short_echo_time == float(fieldmap_info['EchoTime1']):
            magnitude_file = magnitude[0]
        elif short_echo_time == float(fieldmap_info['EchoTime2']):
            magnitude_file = magnitude[1]

        return (short_echo_time, long_echo_time), magnitude_file

    def get_preprocessing_sub_workflow(self):
        """
        Create the second part of the preprocessing workflow.

        Returns:
            - preprocessing : nipype.WorkFlow
        """
        # Infosource Node - To iterate on subjects
        information_source = Node(IdentityInterface(
            fields = ['subject_id', 'run_id']),
            name = 'information_source')
        information_source.iterables = [
            ('subject_id', self.subject_list), ('run_id', self.run_list)
            ]

        # SelectFiles Node - Select necessary files
        templates = {
            'anat' : join('sub-{subject_id}', 'anat', 'sub-{subject_id}_T1w.nii.gz'),
            'func' : join('sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold.nii.gz'),
            'magnitude' : join('sub-{subject_id}', 'fmap', 'sub-{subject_id}_magnitude*.nii.gz'),
            'phasediff' : join('sub-{subject_id}', 'fmap', 'sub-{subject_id}_phasediff.nii.gz'),
            'info_fmap' : join('sub-{subject_id}', 'fmap', 'sub-{subject_id}_phasediff.json'),
            'dartel_flow_field' : join(self.directories.output_dir, 'dartel_template',
                'u_rc1subject_id_{subject_id}_struct_template.nii'),
            'dartel_template' :join(self.directories.output_dir, 'dartel_template', 'template_6.nii')
        }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir

        # Gunzip nodes - gunzip files because SPM do not use .nii.gz files
        gunzip_anat = Node(Gunzip(), name = 'gunzip_anat')
        gunzip_func = Node(Gunzip(), name = 'gunzip_func')
        gunzip_magnitude = Node(Gunzip(), name = 'gunzip_magnitude')
        gunzip_phasediff = Node(Gunzip(), name = 'gunzip_phasediff')

        # Function Node get_fieldmap_info -
        fieldmap_info = Node(Function(
            function = self.get_fieldmap_info,
            input_names = ['fieldmap_info', 'magnitude'],
            output_names = ['echo_times', 'magnitude_file']),
            name = 'fieldmap_info')

        # FieldMap Node -
        fieldmap = Node(FieldMap(), name = 'fieldmap')
        fieldmap.inputs.blip_direction = -1
        fieldmap.inputs.total_readout_time = TaskInformation()['TotalReadoutTime']

        # Get SPM Tissue Probability Maps file
        spm_tissues_file = join(SPMInfo.getinfo()['path'], 'tpm', 'TPM.nii')

        # Segmentation Node - SPM Segment function via custom scripts (defaults left in place)
        segmentation = Node(NewSegment(), name = 'segmentation')
        segmentation.inputs.write_deformation_fields = [True, True]
        segmentation.inputs.channel_info = (0.0001, 60, (True, True))
        segmentation.inputs.tissues = [
            [(spm_tissues_file, 1), 1, (True,False), (True, False)],
            [(spm_tissues_file, 2), 1, (True,False), (True, False)],
            [(spm_tissues_file, 3), 2, (True,False), (False, False)],
            [(spm_tissues_file, 4), 3, (True,False), (False, False)],
            [(spm_tissues_file, 5), 4, (True,False), (False, False)],
            [(spm_tissues_file, 6), 2, (False,False), (False, False)]
        ]

        # Slice timing - SPM slice time correction with default parameters
        slice_timing = Node(SliceTiming(), name = 'slice_timing')
        slice_timing.inputs.num_slices = TaskInformation()['NumberOfSlices']
        slice_timing.inputs.ref_slice = 2
        slice_timing.inputs.slice_order = TaskInformation()['SliceTiming']
        slice_timing.inputs.time_acquisition = TaskInformation()['AcquisitionTime']
        slice_timing.inputs.time_repetition = TaskInformation()['RepetitionTime']

        # Motion correction - SPM realign and unwarp
        motion_correction = Node(RealignUnwarp(), name = 'motion_correction')
        motion_correction.inputs.interp = 4

        # Intrasubject coregistration
        extract_first = Node(ExtractROI(), name = 'extract_first')
        extract_first.inputs.t_min = 1
        extract_first.inputs.t_size = 1
        extract_first.inputs.output_type = 'NIFTI'

        coregistration = Node(Coregister(), name = 'coregistration')
        coregistration.inputs.cost_function = 'nmi'
        coregistration.inputs.jobtype = 'estimate'

        dartel_norm_func = Node(DARTELNorm2MNI(), name = 'dartel_norm_func')
        dartel_norm_func.inputs.fwhm = self.fwhm
        dartel_norm_func.inputs.modulate = False
        dartel_norm_func.inputs.voxel_size = (2.3, 2.3, 2.15)

        dartel_norm_anat = Node(DARTELNorm2MNI(), name = 'dartel_norm_anat')
        dartel_norm_anat.inputs.fwhm = self.fwhm
        dartel_norm_anat.inputs.voxel_size = (1, 1, 1)

        # DataSink Node - store the wanted results in the wanted repository
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir

        # Create preprocessing workflow and connect its nodes
        preprocessing =  Workflow(base_dir = self.directories.working_dir, name = 'preprocessing')
        preprocessing.connect([
            (information_source, select_files, [
                ('subject_id', 'subject_id'),
                ('run_id', 'run_id')]),
            (select_files, gunzip_anat, [('anat', 'in_file')]),
            (select_files, gunzip_func, [('func', 'in_file')]),
            (select_files, gunzip_phasediff, [('phasediff', 'in_file')]),
            (select_files, fieldmap_info, [
                ('info_fmap', 'fieldmap_info'),
                ('magnitude', 'magnitude')]),
            (fieldmap_info, gunzip_magnitude, [('magnitude_file', 'in_file')]),
            (fieldmap_info, fieldmap, [('echo_times', 'echo_times')]),
            (gunzip_magnitude, fieldmap, [('out_file', 'magnitude_file')]),
            (gunzip_phasediff, fieldmap, [('out_file', 'phase_file')]),
            (gunzip_func, fieldmap, [('out_file', 'epi_file')]),
            (fieldmap, motion_correction, [('vdm', 'phase_map')]),
            (gunzip_anat, segmentation, [('out_file', 'channel_files')]),
            (gunzip_func, slice_timing, [('out_file', 'in_files')]),
            (slice_timing, motion_correction, [('timecorrected_files', 'in_files')]),
            (motion_correction, coregistration, [('realigned_unwarped_files', 'apply_to_files')]),
            (gunzip_anat, coregistration, [('out_file', 'target')]),
            (motion_correction, extract_first, [('realigned_unwarped_files', 'in_file')]),
            (extract_first, coregistration, [('roi_file', 'source')]),
            (select_files, dartel_norm_func, [
                ('dartel_flow_field', 'flowfield_files'),
                ('dartel_template', 'template_file')]),
            (select_files, dartel_norm_anat, [
                ('dartel_flow_field', 'flowfield_files'),
                ('dartel_template', 'template_file')]),
            (gunzip_anat, dartel_norm_anat, [('out_file', 'apply_to_files')]),
            (coregistration, dartel_norm_func, [('coregistered_files', 'apply_to_files')]),
            (dartel_norm_func, data_sink, [
                ('normalized_files', 'preprocessing.@normalized_files')]),
            (motion_correction, data_sink, [
                ('realigned_unwarped_files', 'preprocessing.@motion_corrected'),
                ('realignment_parameters', 'preprocessing.@param')]),
            (segmentation, data_sink, [('normalized_class_images', 'preprocessing.@seg')]),
        ])

        # Remove large files, if requested
        if Configuration()['pipelines']['remove_unused_data']:

            # Merge Node - Merge file names to be removed after datasink node is performed
            merge_removable_files = Node(Merge(5), name = 'merge_removable_files')
            merge_removable_files.inputs.ravel_inputs = True

            # Function Nodes remove_files - Remove sizeable files once they aren't needed
            remove_after_datasink = MapNode(Function(
                function = remove_parent_directory,
                input_names = ['_', 'file_name'],
                output_names = []
                ), name = 'remove_after_datasink', iterfield = 'file_name')

            # Add connections
            preprocessing.connect([
                (gunzip_func, merge_removable_files, [('out_file', 'in1')]),
                (gunzip_phasediff, merge_removable_files, [('out_file', 'in2')]),
                (gunzip_magnitude, merge_removable_files, [('out_file', 'in3')]),
                (fieldmap, merge_removable_files, [('vdm', 'in4')]),
                (slice_timing, merge_removable_files, [('timecorrected_files', 'in5')]),
                (merge_removable_files, remove_after_datasink, [('out', 'file_name')]),
                (data_sink, remove_after_datasink, [('out_file', '_')])
            ])

        return preprocessing

    def get_preprocessing(self):
        """
        Create the full preprocessing workflow.

        Returns: a list of nipype.WorkFlow
        """
        return [
            self.get_dartel_template_sub_workflow(),
            self.get_preprocessing_sub_workflow()
        ]

    def get_preprocessing_outputs(self):
        """ Return the names of the files the preprocessing is supposed to generate. """
        return [] # TODO

    def get_run_level_analysis(self):
        """ No run level analysis has been done by team 98BT """
        return None

    def get_parameters_files(parameters_files, wc2_file, motion_corrected_files, subject_id, working_dir):
        """
        Create new tsv files with only desired parameters per subject per run.

        Parameters :
        - parameters_files : paths to subject parameters file (i.e. one per run)
        - wc2_file : path to the wc2 file
        - motion_corrected_files :
        - subject_id : subject for whom the 1st level analysis is made
        - working_dir : directory where to store the parameters files

        Return :
        - parameters_file : paths to new files containing only desired parameters.
        """
        from os import makedirs
        from os.path import join
        from nibabel import load
        from numpy import mean
        from pandas import read_table
        from nilearn.image import iter_img, resample_to_img, resample_img

        from warnings import simplefilter
        # ignore all future warnings
        simplefilter(action = 'ignore', category = FutureWarning)
        simplefilter(action = 'ignore', category = UserWarning)
        simplefilter(action = 'ignore', category = RuntimeWarning)

        # Load wc2 file and create a mask out of it
        wc2 = load(wc2_file)
        wc2_mask = wc2.get_fdata() > 0.6
        wc2_mask = wc2_mask.astype(int)

        mean_wm = [[] for i in range(len(motion_corrected_files))]
        for file_id, file in enumerate(sorted(motion_corrected_files)):
            functional = load(file)

            for slices in iter_img(functional):
                slice_img = resample_to_img(slices, wc2, interpolation='nearest', clip = True)

                slice_data = slice_img.get_fdata()
                masked_slice = slice_data * wc2_mask
                mean_wm[file_id].append(mean(masked_slice))

        parameters_files = []
        for file_id, file in enumerate(sorted(parameters_files)):
            data_frame = read_table(file, sep = '  ', header = None)
            data_frame['Mean_WM'] = mean_wm[file_id]

            new_path = join(working_dir, 'parameters_file',
                f'parameters_file_sub-{subject_id}_run-{str(file_id + 1).zfill(2)}.tsv')

            makedirs(join(working_dir, 'parameters_file'), exist_ok = True)

            with open(new_path, 'w') as writer:
                writer.write(data_frame.to_csv(
                    sep = '\t', index = False, header = False, na_rep = '0.0'))

            parameters_files.append(new_path)

        return parameters_files

    def get_subject_information(event_file: str):
        """
        Create Bunch for specifySPMModel.

        Parameters :
        - event_file: str, events file for a run of a subject

        Returns :
        - subject_info : Bunch corresponding to the event file
        """
        from nipype.interfaces.base import Bunch

        condition_names = ['gamble']

        # Create empty lists
        onset = []
        duration = []
        weights_gain = []
        weights_loss = []
        answers = []

        # Parse event file
        with open(event_file, 'rt') as file:
            next(file)  # skip the header

            for line in file:
                info = line.strip().split()

                onsets.append(float(info[0]))
                durations.append(4.0)
                weights_gain.append(float(info[2]))
                weights_loss.append(float(info[3]))
                if 'accept' in str(info[5]):
                    answers.append(1)
                else:
                    answers.append(0)

        # Create Bunch
        return Bunch(
            conditions = ['gamble'],
            onsets = onsets,
            durations = durations,
            amplitudes = None,
            tmod = None,
            pmod = [
                Bunch(
                    name = ['gain', 'loss', 'answer'], # TODO : warning here : former names must be kept ?
                    poly = [1, 1, 1],
                    param = [weights_gain, weights_loss, answers]
                )
            ],
            regressor_names = None,
            regressors = None
        )

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

        # SelectFiles Node - to select necessary files
        templates = {
            'func' : join(self.directories.output_dir,
                'preprocessing', '_run_id_*_subject_id_{subject_id}',
                'swuasub-{subject_id}_task-MGT_run-*_bold.nii'),
            'motion_correction': join(self.directories.output_dir,
                'preprocessing', '_run_id_*_subject_id_{subject_id}',
                'uasub-{subject_id}_task-MGT_run-*_bold.nii'),
            'param' : join(self.directories.output_dir,
                'preprocessing', '_run_id_*_subject_id_{subject_id}',
                'rp_asub-{subject_id}_task-MGT_run-*_bold.txt'),
            'wc2' : join(self.directories.output_dir,
                'preprocessing', '_run_id_01_subject_id_{subject_id}',
                'wc2sub-{subject_id}_T1w.nii'),
            'events' : join(self.directories.dataset_dir,
                'sub-{subject_id}', 'func', 'sub-{subject_id}_task-MGT_run-*_events.tsv')
        }
        select_files = Node(SelectFiles(templates),name = 'select_files')
        select_files.inputs.base_directory = self.directories.results_dir

        # DataSink Node - store the wanted results in the wanted repository
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir

        # Get Subject Info - get subject specific condition information
        subject_information = MapNode(Function(
            function = self.get_subject_information),
            input_names = ['event_file'],
            output_names = ['subject_information'],
            name = 'subject_information', iterfield = 'event_file')

        # Get parameters
        parameters = Node(Function(
            function = self.get_parameters_files,
            input_names = [
                'parameters_files',
                'wc2_file',
                'motion_corrected_files',
                'subject_id',
                'working_dir'
                ],
            output_names = ['new_parameters_files']),
            name = 'parameters')
        parameters.inputs.working_dir = self.directories.working_dir

        # SpecifyModel - Generates SPM-specific Model
        specify_model = Node(SpecifySPMModel(), name = 'specify_model')
        specify_model.inputs.concatenate_runs = False
        specify_model.inputs.input_units = 'secs'
        specify_model.inputs.output_units = 'secs'
        specify_model.inputs.time_repetition = TaskInformation()['RepetitionTime']
        specify_model.inputs.high_pass_filter_cutoff = 128

        # Level1Design - Generates an SPM design matrix
        model_design = Node(Level1Design(), name = 'model_design')
        model_design.inputs.bases = {'hrf': {'derivs': [1, 1]}}
        model_design.inputs.timing_units = 'secs'
        model_design.inputs.interscan_interval = TaskInformation()['RepetitionTime']

        # EstimateModel - estimate the parameters of the model
        model_estimate = Node(EstimateModel(), name = 'model_estimate')
        model_estimate.inputs.estimation_method = {'Classical': 1}

        # EstimateContrast - estimates contrasts
        contrast_estimate = Node(EstimateContrast(), name = 'contrast_estimate')
        contrast_estimate.inputs.contrasts = self.subject_level_contrasts

        # Create l1 analysis workflow and connect its nodes
        subject_level_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = 'subject_level_analysis'
            )
        subject_level_analysis.connect([
            (information_source, select_files, [('subject_id', 'subject_id')]),
            (information_source, parameters, [('subject_id', 'subject_id')]),
            (select_files, subject_information, [('events', 'event_file')]),
            (select_files, parameters, [
                ('motion_correction', 'motion_corrected_files'),
                ('param', 'parameters_files'),
                ('wc2', 'wc2_file')]),
            (select_files, specify_model, [('func', 'functional_runs')]),
            (subject_information, specify_model, [('subject_information', 'subject_info')]),
            (parameters, specify_model, [
                ('new_parameters_files', 'realignment_parameters')]),
            (specify_model, model_design, [('session_info', 'session_info')]),
            (model_design, model_estimate, [('spm_mat_file', 'spm_mat_file')]),
            (model_estimate, contrast_estimate, [
                ('spm_mat_file', 'spm_mat_file'),
                ('beta_images', 'beta_images'),
                ('residual_image', 'residual_image')]),
            (contrast_estimate, data_sink, [
                ('con_images', 'subject_level_analysis.@con_images'),
                ('spmT_images', 'subject_level_analysis.@spmT_images'),
                ('spm_mat_file', 'subject_level_analysis.@spm_mat_file')])
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
        Returns the 2nd level of analysis workflow.

        Parameters:
            - method: one of "equalRange", "equalIndifference" or "groupComp"

        Returns:
            - group_level_analysis: Nipype WorkFlow
        """
        # Compute the number of participants used to do the analysis
        nb_subjects = len(self.subject_list)

        # Infosource - a function free node to iterate over the list of subject names
        information_source = Node(
            IdentityInterface(
                fields = ['contrast_id', 'subjects']),
                name = 'information_source')
        information_source.iterables = [('contrast_id', self.contrast_list)]

        # SelectFiles
        templates = {
            'contrasts' : join(self.directories.output_dir,
                'subject_level_analysis', '_subject_id_*', 'con_{contrast_id}.nii')
        }

        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.results_dir
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
        threshold = MapNode(Threshold(),
            name = 'threshold', iterfield = ['stat_image', 'contrast_index'])
        threshold.inputs.use_fwe_correction = False
        threshold.inputs.height_threshold = 0.01
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
            (threshold, data_sink, [
                ('thresholded_map', f'group_level_analysis_{method}_nsub_{nb_subjects}.@thresh')]),
            (estimate_model, data_sink, [
                ('mask_image', f'group_level_analysis_{method}_nsub_{nb_subjects}.@mask')]),
            (estimate_contrast, data_sink, [
                ('spm_mat_file', f'group_level_analysis_{method}_nsub_{nb_subjects}.@spm_mat'),
                ('spmT_images', f'group_level_analysis_{method}_nsub_{nb_subjects}.@T'),
                ('con_images', f'group_level_analysis_{method}_nsub_{nb_subjects}.@con')])])

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
                (get_equal_range_subjects, get_contrasts, [('out_list', 'elements')])
                ])

        elif method == 'equalIndifference':
            group_level_analysis.connect([
                (get_equal_indifference_subjects, get_contrasts, [('out_list', 'elements')])
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

            # Node for the design matrix
            two_sample_t_test_design = Node(TwoSampleTTestDesign(),
                name = 'two_sample_t_test_design')

            group_level_analysis.connect([
                (select_files, get_contrasts_2, [('contrasts', 'input_str')]),
                (get_equal_range_subjects, get_contrasts, [('out_list', 'elements')]),
                (get_equal_indifference_subjects, get_contrasts_2, [('out_list', 'elements')]),
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
        """ Return all hypotheses output file names. """
        nb_sub = len(self.subject_list)
        files = [
            # Hypothesis 1
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001', 'spmT_0001.nii'),
            # Hypothesis 2
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001', 'spmT_0001.nii'),
            # Hypothesis 3
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001', 'spmT_0001.nii'),
            # Hypothesis 4
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001', 'spmT_0001.nii'),
            # Hypothesis 5
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold0', 'spmT_0002_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0002.nii'),
            # Hypothesis 6
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold1', 'spmT_0002_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0002.nii'),
            # Hypothesis 7
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0001.nii'),
            # Hypothesis 8
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold1', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0001.nii'),
            # Hypothesis 9
            join(f'group_level_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0001.nii')
        ]
        return [join(self.directories.output_dir, f) for f in files]
