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
from nipype.algorithms.modelgen import SpecifySPMModel
from niflow.nipype1.workflows.fmri.spm import create_DARTEL_template

from narps_open.pipelines import Pipeline
from narps_open.data.task import TaskInformation
from narps_open.data.participants import get_group
from narps_open.core.common import (
    remove_parent_directory, list_intersection, elements_in_string, clean_list
    )
from narps_open.core.image import get_image_timepoints
from narps_open.utils.configuration import Configuration

class PipelineTeam98BT(Pipeline):
    """ A class that defines the pipeline of team 98BT. """

    def __init__(self):
        super().__init__()
        self.fwhm = 8.0
        self.team_id = '98BT'
        self.contrast_list = ['0001', '0002', '0003', '0004']

        # Define contrasts
        gain_conditions = [f'gamble_run{r}xgain^1' for r in range(1,len(self.run_list) + 1)]
        loss_conditions = [f'gamble_run{r}xloss^1' for r in range(1,len(self.run_list) + 1)]
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

        # Init workflow
        dartel_workflow = Workflow(
            base_dir = self.directories.working_dir, name = 'dartel_workflow')

        # IDENTITY INTERFACE - To iterate on subjects
        information_source = Node(IdentityInterface(
            fields = ['subject_id']),
            name = 'information_source')
        information_source.iterables = ('subject_id', self.subject_list)

        # SELECT FILES - Select anat file
        template = {
            'anat' : join('sub-{subject_id}', 'anat', 'sub-{subject_id}_T1w.nii.gz')
        }
        select_files = Node(SelectFiles(template), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir
        dartel_workflow.connect(information_source, 'subject_id', select_files, 'subject_id')

        # GUNZIP - SPM do not use .nii.gz files
        gunzip_anat = Node(Gunzip(), name = 'gunzip_anat')
        dartel_workflow.connect(select_files, 'anat', gunzip_anat, 'in_file')

        # IDENTITY INTERFACE - Join all gunziped files
        dartel_inputs = JoinNode(IdentityInterface(fields = ['structural_files']),
            name = 'dartel_inputs',
            joinsource = 'information_source',
            joinfield = 'structural_files')
        dartel_workflow.connect(gunzip_anat, 'out_file', dartel_inputs, 'structural_files')

        # RENAME - Rename files before dartel workflow
        rename_dartel = MapNode(Rename(format_string = 'subject_id_%(subject_id)s_struct'),
            iterfield = ['in_file', 'subject_id'],
            name = 'rename_dartel')
        rename_dartel.inputs.subject_id = self.subject_list
        rename_dartel.inputs.keep_ext = True
        dartel_workflow.connect(dartel_inputs, 'structural_files', rename_dartel, 'in_file')

        # DARTEL - Using a already existing workflow
        dartel_sub_workflow = create_DARTEL_template(name = 'dartel_sub_workflow')
        dartel_sub_workflow.inputs.inputspec.template_prefix = 'template'
        dartel_workflow.connect(
            rename_dartel, 'out_file', dartel_sub_workflow, 'inputspec.structural_files')

        # DATASINK - store the wanted results in the wanted repository
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir
        dartel_workflow.connect(
            dartel_sub_workflow, 'outputspec.template_file', data_sink, 'dartel_template.@template_file')
        dartel_workflow.connect(
            dartel_sub_workflow, 'outputspec.flow_fields', data_sink, 'dartel_template.@flow_fields')

        # Remove large files, if requested
        if Configuration()['pipelines']['remove_unused_data']:

            # Function Nodes remove_parent_directory - Remove gunziped files
            remove_gunzip = Node(Function(
                function = remove_parent_directory,
                input_names = ['_', 'file_name'],
                output_names = []
                ), name = 'remove_gunzip')
            dartel_workflow.connect(gunzip_anat, 'out_file', remove_gunzip, 'file_name')
            dartel_workflow.connect(data_sink, 'out_file', remove_gunzip, '_')

        return dartel_workflow

    def get_fieldmap_info(fieldmap_info_file, magnitude_files):
        """
        Function to get information necessary to compute the fieldmap.

        Parameters:
            - fieldmap_info_file: str, file with fieldmap information
            - magnitude_files: list of str, list of magnitude files

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
            magnitude_file = magnitude_files[0]
        elif short_echo_time == float(fieldmap_info['EchoTime2']):
            magnitude_file = magnitude_files[1]

        return (short_echo_time, long_echo_time), magnitude_file

    def get_preprocessing_sub_workflow(self):
        """
        Create the second part of the preprocessing workflow.

        Returns:
            - preprocessing : nipype.WorkFlow
        """
        # Create preprocessing workflow
        preprocessing =  Workflow(base_dir = self.directories.working_dir, name = 'preprocessing')

        # IDENTITY INTERFACE - To iterate on subjects
        information_source_subjects = Node(IdentityInterface(
            fields = ['subject_id']),
            name = 'information_source_subjects')
        information_source_subjects.iterables = ('subject_id', self.subject_list)

        # IDENTITY INTERFACE - To iterate on runs
        information_source_runs = Node(IdentityInterface(
            fields = ['subject_id', 'run_id']),
            name = 'information_source_runs')
        information_source_runs.iterables = ('run_id', self.run_list)
        preprocessing.connect(
            information_source_subjects, 'subject_id', information_source_runs, 'subject_id')

        # SELECT FILES - Select necessary subject files
        templates = {
            'anat' : join('sub-{subject_id}', 'anat', 'sub-{subject_id}_T1w.nii.gz'),
            'magnitude' : join('sub-{subject_id}', 'fmap', 'sub-{subject_id}_magnitude*.nii.gz'),
            'phasediff' : join('sub-{subject_id}', 'fmap', 'sub-{subject_id}_phasediff.nii.gz'),
            'info_fmap' : join('sub-{subject_id}', 'fmap', 'sub-{subject_id}_phasediff.json'),
            'dartel_flow_field' : join(self.directories.output_dir, 'dartel_template',
                'u_rc1subject_id_{subject_id}_struct_template.nii'),
            'dartel_template' :join(
                self.directories.output_dir, 'dartel_template', 'template_6.nii')
        }
        select_subject_files = Node(SelectFiles(templates), name = 'select_subject_files')
        select_subject_files.inputs.base_directory = self.directories.dataset_dir
        preprocessing.connect(
            information_source_subjects, 'subject_id', select_subject_files, 'subject_id')

        # SELECT FILES - Select necessary run files
        templates = {
            'func' : join('sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold.nii.gz')
        }
        select_run_files = Node(SelectFiles(templates), name = 'select_run_files')
        select_run_files.inputs.base_directory = self.directories.dataset_dir
        preprocessing.connect(information_source_runs, 'subject_id', select_run_files, 'subject_id')
        preprocessing.connect(information_source_runs, 'run_id', select_run_files, 'run_id')

        # GUNZIP - gunzip files because SPM do not use .nii.gz files
        gunzip_anat = Node(Gunzip(), name = 'gunzip_anat')
        gunzip_func = Node(Gunzip(), name = 'gunzip_func')
        gunzip_magnitude = Node(Gunzip(), name = 'gunzip_magnitude')
        gunzip_phasediff = Node(Gunzip(), name = 'gunzip_phasediff')
        preprocessing.connect(select_subject_files, 'anat', gunzip_anat, 'in_file'),
        preprocessing.connect(select_run_files, 'func', gunzip_func, 'in_file')
        preprocessing.connect(select_subject_files, 'phasediff', gunzip_phasediff, 'in_file')

        # FUNCTION Node get_fieldmap_info - Retrieve magnitude and phasediff metadata to decide
        # which files to use for the fieldmap node, and what echo times
        fieldmap_info = Node(Function(
            function = self.get_fieldmap_info,
            input_names = ['fieldmap_info_file', 'magnitude_files'],
            output_names = ['echo_times', 'magnitude_file']),
            name = 'fieldmap_info')
        preprocessing.connect(
            select_subject_files, 'info_fmap', fieldmap_info, 'fieldmap_info_file')
        preprocessing.connect(select_subject_files, 'magnitude', fieldmap_info, 'magnitude_files')
        preprocessing.connect(fieldmap_info, 'magnitude_file', gunzip_magnitude, 'in_file')

        # FIELDMAP Node - create fieldmap from phasediff and magnitude files
        fieldmap = Node(FieldMap(), name = 'fieldmap')
        fieldmap.inputs.blip_direction = -1
        fieldmap.inputs.total_readout_time = TaskInformation()['TotalReadoutTime']
        preprocessing.connect(fieldmap_info, 'echo_times', fieldmap, 'echo_times')
        preprocessing.connect(gunzip_magnitude, 'out_file', fieldmap, 'magnitude_file')
        preprocessing.connect(gunzip_phasediff, 'out_file', fieldmap, 'phase_file')
        preprocessing.connect(gunzip_func, 'out_file', fieldmap, 'epi_file')

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
        preprocessing.connect(gunzip_anat, 'out_file', segmentation, 'channel_files')

        # Slice timing - SPM slice time correction with default parameters
        slice_timing = Node(SliceTiming(), name = 'slice_timing')
        slice_timing.inputs.num_slices = TaskInformation()['NumberOfSlices']
        slice_timing.inputs.ref_slice = 2
        slice_timing.inputs.slice_order = TaskInformation()['SliceTiming']
        slice_timing.inputs.time_acquisition = TaskInformation()['AcquisitionTime']
        slice_timing.inputs.time_repetition = TaskInformation()['RepetitionTime']
        preprocessing.connect(gunzip_func, 'out_file', slice_timing, 'in_files')

        # Motion correction - SPM realign and unwarp
        motion_correction = Node(RealignUnwarp(), name = 'motion_correction')
        motion_correction.inputs.interp = 4
        preprocessing.connect(fieldmap, 'vdm', motion_correction, 'phase_map')
        preprocessing.connect(slice_timing, 'timecorrected_files', motion_correction, 'in_files')

        # Intrasubject coregistration
        extract_first = Node(Function(
            function = get_image_timepoints,
            input_names = ['in_file', 'start_time_point', 'end_time_point'],
            output_names = ['roi_file']
            ), name = 'extract_first')
        extract_first.inputs.start_time_point = 1
        extract_first.inputs.end_time_point = 1
        preprocessing.connect(
            motion_correction, 'realigned_unwarped_files', extract_first, 'in_file')

        coregistration = Node(Coregister(), name = 'coregistration')
        coregistration.inputs.cost_function = 'nmi'
        coregistration.inputs.jobtype = 'estimate'
        preprocessing.connect(
            motion_correction, 'realigned_unwarped_files', coregistration, 'apply_to_files')
        preprocessing.connect(gunzip_anat, 'out_file', coregistration, 'target')
        preprocessing.connect(extract_first, 'roi_file', coregistration, 'source')

        dartel_norm_func = Node(DARTELNorm2MNI(), name = 'dartel_norm_func')
        dartel_norm_func.inputs.fwhm = self.fwhm
        dartel_norm_func.inputs.modulate = False
        dartel_norm_func.inputs.voxel_size = (2.3, 2.3, 2.15)
        preprocessing.connect(
            select_subject_files, 'dartel_flow_field', dartel_norm_func, 'flowfield_files')
        preprocessing.connect(
            select_subject_files, 'dartel_template',  dartel_norm_func, 'template_file')
        preprocessing.connect(
            coregistration, 'coregistered_files', dartel_norm_func, 'apply_to_files')

        dartel_norm_anat = Node(DARTELNorm2MNI(), name = 'dartel_norm_anat')
        dartel_norm_anat.inputs.fwhm = self.fwhm
        dartel_norm_anat.inputs.voxel_size = (1, 1, 1)
        preprocessing.connect(
            select_subject_files, 'dartel_flow_field', dartel_norm_anat, 'flowfield_files')
        preprocessing.connect(
            select_subject_files, 'dartel_template', dartel_norm_anat, 'template_file')
        preprocessing.connect(gunzip_anat, 'out_file', dartel_norm_anat, 'apply_to_files')

        # DataSink Node - store the wanted results in the wanted repository
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir
        preprocessing.connect(
            dartel_norm_func, 'normalized_files', data_sink, 'preprocessing.@normalized_files')
        preprocessing.connect(
            motion_correction, 'realigned_unwarped_files',
            data_sink, 'preprocessing.@motion_corrected')
        preprocessing.connect(
            motion_correction, 'realignment_parameters', data_sink, 'preprocessing.@param')
        preprocessing.connect(
            segmentation, 'normalized_class_images', data_sink, 'preprocessing.@seg')

        # Remove large files, if requested
        if Configuration()['pipelines']['remove_unused_data']:

            # Merge Node - Merge anat file names to be removed after datasink node is performed
            merge_removable_anat_files = Node(Merge(3), name = 'merge_removable_anat_files')
            merge_removable_anat_files.inputs.ravel_inputs = True

            # Function Nodes remove_files - Remove sizeable anat files once they aren't needed
            remove_anat_after_datasink = MapNode(Function(
                function = remove_parent_directory,
                input_names = ['_', 'file_name'],
                output_names = []
                ), name = 'remove_anat_after_datasink', iterfield = 'file_name')
            preprocessing.connect([
                (gunzip_phasediff, merge_removable_anat_files, [('out_file', 'in1')]),
                (gunzip_magnitude, merge_removable_anat_files, [('out_file', 'in2')]),
                (fieldmap, merge_removable_anat_files, [('vdm', 'in3')]),
                (merge_removable_anat_files, remove_anat_after_datasink, [('out', 'file_name')]),
                (data_sink, remove_anat_after_datasink, [('out_file', '_')])
            ])

            # Merge Node - Merge func file names to be removed after datasink node is performed
            merge_removable_func_files = Node(Merge(2), name = 'merge_removable_func_files')
            merge_removable_func_files.inputs.ravel_inputs = True

            # Function Nodes remove_files - Remove sizeable func files once they aren't needed
            remove_func_after_datasink = MapNode(Function(
                function = remove_parent_directory,
                input_names = ['_', 'file_name'],
                output_names = []
                ), name = 'remove_func_after_datasink', iterfield = 'file_name')
            preprocessing.connect([
                (gunzip_func, merge_removable_func_files, [('out_file', 'in1')]),
                (slice_timing, merge_removable_func_files, [('timecorrected_files', 'in2')]),
                (merge_removable_func_files, remove_func_after_datasink, [('out', 'file_name')]),
                (data_sink, remove_func_after_datasink, [('out_file', '_')])
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

        # Outputs from dartel workflow
        return_list = [join(self.directories.output_dir, 'dartel_template', 'template_6.nii')]
        return_list += [join(self.directories.output_dir, 'dartel_template',
            f'u_rc1subject_id_{subject_id}_struct_template.nii')\
            for subject_id in self.subject_list]

        # Outputs from preprocessing
        parameters = {
            'subject_id': self.subject_list,
            'run_id': self.run_list,
        }
        parameter_sets = product(*parameters.values())
        output_dir = join(self.directories.output_dir, 'preprocessing', '_subject_id_{subject_id}')
        run_output_dir = join(output_dir, '_run_id_{run_id}')
        templates = [
            # Realignment parameters
            join(run_output_dir, 'rp_asub-{subject_id}_task-MGT_run-{run_id}_bold.txt'),
            # Realigned unwarped files
            join(run_output_dir, 'uasub-{subject_id}_task-MGT_run-{run_id}_bold.nii'),
            # Normalized_files
            join(run_output_dir, 'swuasub-{subject_id}_task-MGT_run-{run_id}_bold.nii'),
            # Normalized class images
            join(output_dir, 'wc2sub-{subject_id}_T1w.nii')
        ]
        return_list += [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets for template in templates]

        return return_list

    def get_run_level_analysis(self):
        """ No run level analysis has been done by team 98BT """
        return None

    def get_parameters_file(
        parameters_file: str,
        wc2_file: str,
        func_file: str, subject_id: str, run_id: str, working_dir: str):
        """
        Create a new tsv file, by adding the mean signal in white matter to other parameters
            in parameters_file.

        Parameters :
        - parameters_file : path to subject parameters file (i.e. one per run)
        - wc2_file : path to the segmented white matter file
        - func_file : path to the functional data
        - run_id : run for which the analysis is made
        - subject_id : subject for whom the analysis is made
        - working_dir : directory where to store the output files

        Return :
        - path to new file containing only desired parameters.
        """
        from os import makedirs
        from os.path import join
        from nibabel import load
        from nibabel.processing import resample_from_to
        from numpy import mean
        from pandas import read_csv

        # Ignore all future warnings
        from warnings import simplefilter
        simplefilter(action = 'ignore', category = FutureWarning)
        simplefilter(action = 'ignore', category = UserWarning)
        simplefilter(action = 'ignore', category = RuntimeWarning)

        # Load wc2 file and create a mask out of it
        wm_class_image = load(wc2_file)
        wm_mask_data = wm_class_image.get_fdata() > 0.6
        wm_mask_data = wm_mask_data.astype(int)

        # Compute the mean signal in white matter, for each slice of the functional data
        mean_wm = []
        func_image = load(func_file)
        for index in range(func_image.header.get_data_shape()[3]):
            resampled_func = resample_from_to(
                func_image.slicer[..., index], wm_class_image, order = 0, mode = 'nearest')

            # Append mean value of masked data
            mean_wm.append(mean(resampled_func.get_fdata() * wm_mask_data))

        # Create new parameters file
        data_frame = read_csv(parameters_file, sep = '\t', dtype = str)
        data_frame['Mean_WM'] = mean_wm
        new_parameters_file = join(working_dir, 'parameters_files',
            f'parameters_file_sub-{subject_id}_run-{run_id}.tsv')

        makedirs(join(working_dir, 'parameters_files'), exist_ok = True)

        with open(new_parameters_file, 'w') as writer:
            writer.write(data_frame.to_csv(
                sep = '\t', index = False, header = False, na_rep = '0.0'))

        return new_parameters_file

    def get_subject_information(event_file: str, short_run_id: int):
        """
        Create Bunch for specifySPMModel.

        Parameters :
        - event_file: str, events file for a run of a subject
        - short_run_id: str, an identifier for the run corresponding to the event_file
            must be '1' for the first run, '2' for the second run, etc.

        Returns :
        - subject_info : Bunch corresponding to the event file
        """
        from nipype.interfaces.base import Bunch

        # Create empty lists
        onsets = []
        durations = []
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
                if 'weakly_accept' in str(info[5]):
                    answers.append(1)
                elif 'strongly_accept' in str(info[5]):
                    answers.append(2)
                elif 'weakly_reject' in str(info[5]):
                    answers.append(-1)
                else:
                    answers.append(-2)

        # Create Bunch
        return Bunch(
            conditions = [f'gamble_run{short_run_id}'],
            onsets = [onsets],
            durations = [durations],
            amplitudes = None,
            tmod = None,
            pmod = [
                Bunch(
                    name = ['gain', 'loss', 'answers'],
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
        # Init workflow
        subject_level_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = 'subject_level_analysis'
            )

        # IDENTITY INTERFACE - To iterate on subjects
        information_source = Node(IdentityInterface(
            fields = ['subject_id']),
            name = 'information_source')
        information_source.iterables = [('subject_id', self.subject_list)]

        # SELECT FILES - to select necessary files
        templates = {
            'func' : join(self.directories.output_dir,
                'preprocessing', '_subject_id_{subject_id}', '_run_id_*',
                'swuasub-{subject_id}_task-MGT_run-*_bold.nii'),
            'motion_correction': join(self.directories.output_dir,
                'preprocessing', '_subject_id_{subject_id}', '_run_id_*',
                'uasub-{subject_id}_task-MGT_run-*_bold.nii'),
            'param' : join(self.directories.output_dir,
                'preprocessing', '_subject_id_{subject_id}', '_run_id_*',
                'rp_asub-{subject_id}_task-MGT_run-*_bold.txt'),
            'wc2' : join(self.directories.output_dir,
                'preprocessing', '_subject_id_{subject_id}',
                'wc2sub-{subject_id}_T1w.nii'),
            'events' : join(self.directories.dataset_dir,
                'sub-{subject_id}', 'func', 'sub-{subject_id}_task-MGT_run-*_events.tsv')
        }
        select_files = Node(SelectFiles(templates),name = 'select_files')
        select_files.inputs.base_directory = self.directories.results_dir
        subject_level_analysis.connect(
            information_source, 'subject_id', select_files, 'subject_id')

        # FUNCTION node get_subject_information - get subject specific condition information
        subject_information = MapNode(Function(
            function = self.get_subject_information,
            input_names = ['event_file', 'short_run_id'],
            output_names = ['subject_info']),
            name = 'subject_information', iterfield = ['event_file', 'short_run_id'])
        subject_information.inputs.short_run_id = list(range(1, len(self.run_list) + 1))
        subject_level_analysis.connect(select_files, 'events', subject_information, 'event_file')

        # FUNCTION node get_parameters_file - Get subject parameters
        parameters = MapNode(Function(
            function = self.get_parameters_file,
            input_names = [
                'parameters_file',
                'wc2_file',
                'func_file',
                'subject_id',
                'run_id',
                'working_dir'
                ],
            output_names = ['new_parameters_file']),
            iterfield = ['parameters_file', 'func_file'],
            name = 'parameters')
        parameters.inputs.run_id = self.run_list
        parameters.inputs.working_dir = self.directories.working_dir
        subject_level_analysis.connect(information_source, 'subject_id', parameters, 'subject_id')
        subject_level_analysis.connect(select_files, 'motion_correction', parameters, 'func_file')
        subject_level_analysis.connect(select_files, 'param', parameters,'parameters_file')
        subject_level_analysis.connect(select_files, 'wc2', parameters,'wc2_file')

        # SPECIFY MODEL - Generates SPM-specific Model
        specify_model = Node(SpecifySPMModel(), name = 'specify_model')
        specify_model.inputs.concatenate_runs = False
        specify_model.inputs.input_units = 'secs'
        specify_model.inputs.output_units = 'secs'
        specify_model.inputs.time_repetition = TaskInformation()['RepetitionTime']
        specify_model.inputs.high_pass_filter_cutoff = 128
        subject_level_analysis.connect(select_files, 'func', specify_model, 'functional_runs')
        subject_level_analysis.connect(
            subject_information, 'subject_info', specify_model, 'subject_info')
        subject_level_analysis.connect(
            parameters, 'new_parameters_file', specify_model, 'realignment_parameters')

        # LEVEL1DESIGN - Generates an SPM design matrix
        model_design = Node(Level1Design(), name = 'model_design')
        model_design.inputs.bases = {'hrf': {'derivs': [1, 1]}}
        model_design.inputs.timing_units = 'secs'
        model_design.inputs.interscan_interval = TaskInformation()['RepetitionTime']
        model_design.inputs.volterra_expansion_order = 2
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
            model_estimate, 'residual_image', contrast_estimate,'residual_image')

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

        # Init workflow
        group_level_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = f'group_level_analysis_{method}_nsub_{nb_subjects}')

        # IDENTITY INTERFACE - iterate over the list of contrasts
        information_source = Node(
            IdentityInterface(
                fields = ['contrast_id', 'subjects']),
                name = 'information_source')
        information_source.iterables = [('contrast_id', self.contrast_list)]

        # SELECT FILES - get contrast files from subject-level output
        templates = {
            'contrasts' : join(self.directories.output_dir,
                'subject_level_analysis', '_subject_id_*', 'con_{contrast_id}.nii')
        }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.results_dir
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

        # ESTIMATE MODEL (inputs will be set later, depending on the method)
        estimate_model = Node(EstimateModel(), name = 'estimate_model')
        estimate_model.inputs.estimation_method = {'Classical':1}

        # ESTIMATE CONTRAST
        estimate_contrast = Node(EstimateContrast(), name = 'estimate_contrast')
        estimate_contrast.inputs.group_contrast = True
        group_level_analysis.connect(
            estimate_model, 'spm_mat_file', estimate_contrast, 'spm_mat_file')
        group_level_analysis.connect(
            estimate_model, 'residual_image', estimate_contrast, 'residual_image')
        group_level_analysis.connect(
            estimate_model, 'beta_images', estimate_contrast, 'beta_images')

        # THRESHOLD - Create thresholded maps
        threshold = MapNode(Threshold(),
            name = 'threshold', iterfield = ['stat_image', 'contrast_index'])
        threshold.inputs.use_fwe_correction = False
        threshold.inputs.height_threshold = 0.01
        threshold.inputs.extent_fdr_p_threshold = 0.05
        threshold.inputs.use_topo_fdr = False
        threshold.inputs.force_activation = True
        threshold.synchronize = True
        group_level_analysis.connect(estimate_contrast, 'spm_mat_file', threshold, 'spm_mat_file')
        group_level_analysis.connect(estimate_contrast, 'spmT_images', threshold, 'stat_image')

        if method in ('equalRange', 'equalIndifference'):
            estimate_contrast.inputs.contrasts = [
                ('Group', 'T', ['mean'], [1]),
                ('Group', 'T', ['mean'], [-1])
                ]

            # ONE SAMPLE T TEST DESIGN - Create the design matrix
            one_sample_t_test_design = Node(OneSampleTTestDesign(),
                name = 'one_sample_t_test_design')
            group_level_analysis.connect(
                get_contrasts, ('out_list', clean_list), one_sample_t_test_design, 'in_files')
            group_level_analysis.connect(
                one_sample_t_test_design, 'spm_mat_file', estimate_model, 'spm_mat_file')

            threshold.inputs.contrast_index = [1, 2]

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
            group_level_analysis.connect(
                get_equal_range_subjects, ('out_list', complete_subject_ids),
                get_contrasts, 'elements')

            estimate_contrast.inputs.contrasts = [
                ('Eq range vs Eq indiff in loss', 'T', ['Group_{1}', 'Group_{2}'], [1, -1])
                ]

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
            group_level_analysis.connect(select_files, 'contrasts', get_contrasts_2, 'input_str')
            group_level_analysis.connect(
                get_equal_indifference_subjects, ('out_list', complete_subject_ids),
                get_contrasts_2, 'elements')

            # TWO SAMPLE T TEST DESIGN - Create the design matrix
            two_sample_t_test_design = Node(TwoSampleTTestDesign(),
                name = 'two_sample_t_test_design')
            group_level_analysis.connect(
                get_contrasts, ('out_list', clean_list),
                two_sample_t_test_design, 'group1_files')
            group_level_analysis.connect(
                get_contrasts_2, ('out_list', clean_list),
                two_sample_t_test_design, 'group2_files')
            group_level_analysis.connect(
                two_sample_t_test_design, 'spm_mat_file', estimate_model, 'spm_mat_file')

            threshold.inputs.contrast_index = [1]

        # Datasink - save important files
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir
        group_level_analysis.connect(
            threshold, 'thresholded_map',
            data_sink, f'group_level_analysis_{method}_nsub_{nb_subjects}.@thresh')
        group_level_analysis.connect(
            estimate_model, 'mask_image',
            data_sink, f'group_level_analysis_{method}_nsub_{nb_subjects}.@mask')
        group_level_analysis.connect(
            estimate_contrast, 'spm_mat_file',
            data_sink, f'group_level_analysis_{method}_nsub_{nb_subjects}.@spm_mat')
        group_level_analysis.connect(
            estimate_contrast, 'spmT_images',
            data_sink, f'group_level_analysis_{method}_nsub_{nb_subjects}.@T')
        group_level_analysis.connect(
            estimate_contrast, 'con_images',
            data_sink, f'group_level_analysis_{method}_nsub_{nb_subjects}.@con')

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
