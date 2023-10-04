#!/usr/bin/python
# coding: utf-8

""" Write the work of NARPS team 98BT using Nipype """

from os.path import join
from itertools import product

from nipype import Workflow, Node, MapNode, JoinNode
from nipype.interfaces.utility import IdentityInterface, Function, Rename
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.algorithms.misc import Gunzip
from nipype.interfaces.spm import (
    Coregister, OneSampleTTestDesign,
    EstimateModel, EstimateContrast, Level1Design,
    TwoSampleTTestDesign, RealignUnwarp, NewSegment, SliceTiming,
    DARTELNorm2MNI, FieldMap, Threshold
    )
from nipype.interfaces.fsl import ExtractROI
from nipype.algorithms.modelgen import SpecifySPMModel
from niflow.nipype1.workflows.fmri.spm import create_DARTEL_template

from narps_open.pipelines import Pipeline
from narps_open.data.task import Pipeline

class PipelineTeam98BT(Pipeline):
    """ A class that defines the pipeline of team 98BT. """

    def __init__(self):
        super().__init__()
        self.fwhm = 8.0
        self.team_id = '98BT'
        self.contrast_list = ['0001', '0002', '0003', '0004']

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
        infosource_dartel = Node(IdentityInterface(
            fields = ['subject_id']),
            name = 'infosource_dartel')
        infosource_dartel.iterables = ('subject_id', self.subject_list)

        # Templates to select files node
        template = {
            'anat' : join('sub-{subject_id}', 'anat', 'sub-{subject_id}_T1w.nii.gz')
        }

        # SelectFiles node - to select necessary files
        selectfiles_dartel = Node(SelectFiles(template,
            base_directory = self.directories.dataset_dir),
            name = 'selectfiles_dartel')

        # Gunzip node - SPM do not use .nii.gz files
        gunzip_anat = Node(Gunzip(),
            name = 'gunzip_anat')

        def get_dartel_input(structural_files):
            print(structural_files)
            return structural_files

        dartel_input = JoinNode(Function(
            function = get_dartel_input,
            input_names = ['structural_files'],
            output_names = ['structural_files']),
            name = 'dartel_input',
            joinsource = 'infosource_dartel',
            joinfield = 'structural_files')

        rename_dartel = MapNode(Rename(
            format_string = 'subject_id_%(subject_id)s_struct'),
            iterfield = ['in_file', 'subject_id'],
            name = 'rename_dartel')
        rename_dartel.inputs.subject_id = self.subject_list
        rename_dartel.inputs.keep_ext = True

        dartel_workflow = create_DARTEL_template(name = 'dartel_workflow')
        dartel_workflow.inputs.inputspec.template_prefix = 'template'

        # DataSink Node - store the wanted results in the wanted repository
        datasink_dartel = Node(DataSink(base_directory = self.directories.output_dir),
            name='datasink_dartel')

        # Create dartel workflow and connect its nodes
        dartel = Workflow(base_dir = self.directories.working_dir, name = 'dartel')
        dartel.connect([
            (infosource_dartel, selectfiles_dartel, [('subject_id', 'subject_id')]),
            (selectfiles_dartel, gunzip_anat, [('anat', 'in_file')]),
            (gunzip_anat, dartel_input, [('out_file', 'structural_files')]),
            (dartel_input, rename_dartel, [('structural_files', 'in_file')]),
            (rename_dartel, dartel_workflow, [('out_file', 'inputspec.structural_files')]),
            (dartel_workflow, datasink_dartel, [
                ('outputspec.template_file', 'dartel_template.@template_file'),
                ('outputspec.flow_fields', 'dartel_template.@flow_fields')])
            ])

        return dartel

    def remove_temporary_files(_, subject_id, run_id, working_dir):
        """
        This method is used in a Function node to fully remove
        temporary files, once they aren't needed anymore.

        Parameters:
        - _: Node input only used for triggering the Node
        - subject_id: str, subject id from which to remove the files
        - run_id: str, run id of the files to remove
        - working_dir: str, path to the working directory
        """
        from os.path import join
        from shutil import rmtree

        preprocessing_dir = join(working_dir, 'preprocessing',
            f'run_id_{run_id}_subject_id_{subject_id}')

        for directory in [
            'gunzip_func',
            'gunzip_phasediff',
            'fieldmap_infos',
            'gunzip_magnitude',
            'fieldmap',
            'slice_timing']:
            try:
                rmtree(join(preprocessing_dir, directory))
            except OSError as error:
                print(error)
            else:
                print(f'Successfully deleted : {directory}')

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
        infosource = Node(IdentityInterface(
            fields = ['subject_id', 'run_id']),
            name = 'infosource')
        infosource.iterables = [('subject_id', self.subject_list), ('run_id', self.run_list)]

        # Templates to select files node
        templates = {
            'anat' : join('sub-{subject_id}', 'anat', 'sub-{subject_id}_T1w.nii.gz'),
            'func' : join('sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold.nii.gz'),
            'magnitude' : join('sub-{subject_id}', 'fmap', 'sub-{subject_id}_magnitude*.nii.gz'),
            'phasediff' : join('sub-{subject_id}', 'fmap', 'sub-{subject_id}_phasediff.nii.gz'),
            'info_fmap' : join('sub-{subject_id}', 'fmap', 'sub-{subject_id}_phasediff.json'),
            'dartel_flow_field' : join(result_dir, output_dir, 'dartel_template',
                'u_rc1subject_id_{subject_id}_struct_template.nii'),
            'dartel_template' :join(result_dir, output_dir, 'dartel_template', 'template_6.nii')
        }

        # SelectFiles node - to select necessary files
        selectfiles_preproc = Node(SelectFiles(templates,
            base_directory = self.directories.dataset_dir),
            name = 'selectfiles_preproc')

        # Gunzip nodes - gunzip files because SPM do not use .nii.gz files
        gunzip_anat = Node(Gunzip(), name = 'gunzip_anat')
        gunzip_func = Node(Gunzip(), name = 'gunzip_func')
        gunzip_magnitude = Node(Gunzip(), name = 'gunzip_magnitude')
        gunzip_phasediff = Node(Gunzip(), name = 'gunzip_phasediff')

        fieldmap_info = Node(Function(
            function = self.get_fieldmap_info,
            input_names = ['fieldmap_info', 'magnitude'],
            output_names = ['echo_times', 'magnitude_file']),
            name = 'fieldmap_info')

        fieldmap = Node(FieldMap(blip_direction = -1),
            name = 'fieldmap')
        fieldmap.inputs.total_readout_time = self.total_readout_time

        # Segmentation - SPM Segment function via custom scripts (defaults left in place)
        tissue_list = [
            [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 1), 1, (True,False), (True, False)],
            [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 2), 1, (True,False), (True, False)],
            [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 3), 2, (True,False), (False, False)],
            [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 4), 3, (True,False), (False, False)],
            [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 5), 4, (True,False), (False, False)],
            [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 6), 2, (False,False), (False, False)]
        ]

        segmentation = Node(NewSegment(
            write_deformation_fields = [True, True], tissues = tissue_list,
            channel_info = (0.0001, 60, (True, True))),
            name = 'segmentation')

        # Slice timing - SPM slice time correction with default parameters
        slice_timing = Node(SliceTiming(
            num_slices = self.number_of_slices, ref_slice = 2,
            slice_order = self.slice_timing, time_acquisition = self.acquisition_time,
            time_repetition = self.tr),
            name = 'slice_timing')

        # Motion correction - SPM realign and unwarp
        motion_correction = Node(RealignUnwarp(
            interp = 4),
            name = 'motion_correction')

        # Intrasubject coregistration
        extract_first = Node(ExtractROI(
            t_min = 1, t_size = 1, output_type = 'NIFTI'),
            name = 'extract_first')
        coregistration = Node(Coregister(
            cost_function = 'nmi', jobtype = 'estimate'),
            name = 'coregistration')

        dartel_norm_func = Node(DARTELNorm2MNI(
            fwhm = self.fwhm, modulate = False, voxel_size = (2.3, 2.3, 2.15)),
            name = 'dartel_norm_func')

        dartel_norm_anat = Node(DARTELNorm2MNI(
            fwhm = self.fwhm, voxel_size = (1, 1, 1)),
            name = 'dartel_norm_anat')

        # Function node remove_temporary_files - remove temporary files
        remove_temporary_files = Node(Function(
            function = self.remove_temporary_files,
            input_names = ['_', 'subject_id', 'run_id', 'working_dir'],
            output_names = []),
            name = 'remove_temporary_files')
        remove_temporary_files.inputs.working_dir = self.directories.working_dir

        # DataSink Node - store the wanted results in the wanted repository
        datasink_preproc = Node(DataSink(base_directory = self.directories.output_dir),
            name='datasink_preproc')

        # Create preprocessing workflow and connect its nodes
        preprocessing =  Workflow(base_dir = self.directories.working_dir, name = 'preprocessing')
        preprocessing.connect([
            (infosource, selectfiles_preproc, [
                ('subject_id', 'subject_id'),
                ('run_id', 'run_id')]),
            (infosource, remove_temporary_files, [
                ('subject_id', 'subject_id'),
                ('run_id', 'run_id')]),
            (selectfiles_preproc, gunzip_anat, [('anat', 'in_file')]),
            (selectfiles_preproc, gunzip_func, [('func', 'in_file')]),
            (selectfiles_preproc, gunzip_phasediff, [('phasediff', 'in_file')]),
            (selectfiles_preproc, fieldmap_info, [
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
            (motion_correction, remove_temporary_files, [('realigned_unwarped_files', '_')]),
            (motion_correction, coregistration, [('realigned_unwarped_files', 'apply_to_files')]),
            (gunzip_anat, coregistration, [('out_file', 'target')]),
            (motion_correction, extract_first, [('realigned_unwarped_files', 'in_file')]),
            (extract_first, coregistration, [('roi_file', 'source')]),
            (selectfiles_preproc, dartel_norm_func, [
                ('dartel_flow_field', 'flowfield_files'),
                ('dartel_template', 'template_file')]),
            (selectfiles_preproc, dartel_norm_anat, [
                ('dartel_flow_field', 'flowfield_files'),
                ('dartel_template', 'template_file')]),
            (gunzip_anat, dartel_norm_anat, [('out_file', 'apply_to_files')]),
            (coregistration, dartel_norm_func, [('coregistered_files', 'apply_to_files')]),
            (dartel_norm_func, datasink_preproc, [
                ('normalized_files', 'preprocessing.@normalized_files')]),
            (motion_correction, datasink_preproc, [
                ('realigned_unwarped_files', 'preprocessing.@motion_corrected'),
                ('realignment_parameters', 'preprocessing.@param')]),
           (segmentation, datasink_preproc, [('normalized_class_images', 'preprocessing.@seg')])
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


    def get_parameters_files(
        parameters_files, wc2_file, motion_corrected_files, subject_id, working_dir):
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
        simplefilter(action='ignore', category=FutureWarning)
        simplefilter(action='ignore', category=UserWarning)
        simplefilter(action='ignore', category=RuntimeWarning)

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

    def get_subject_info(event_files, runs):
        """
        Create Bunchs for specifySPMModel.

        Parameters :
        - event_files: list of str, list of events files (one per run) for the subject
        - runs: list of str, list of runs to use

        Returns :
        - subject_info : list of Bunch for 1st level analysis.
        """
        from nipype.interfaces.base import Bunch

        condition_names = ['gamble']
        onset = {}
        duration = {}
        weights_gain = {}
        weights_loss = {}
        answers = {}

        for run_id in range(len(runs)):
            # Create dictionary items with empty lists
            onset.update({s + '_run' + str(run_id + 1) : [] for s in condition_names})
            duration.update({s + '_run' + str(run_id + 1) : [] for s in condition_names})
            weights_gain.update({'gain_run' + str(run_id + 1) : []})
            weights_loss.update({'loss_run' + str(run_id + 1) : []})
            answers.update({'answers_run' + str(run_id + 1) : []})

            with open(event_files[run_id], 'rt') as event_file:
                next(event_file)  # skip the header

                for line in event_file:
                    info = line.strip().split()

                    for condition in condition_names:
                        val = condition + '_run' + str(run_id + 1) # trial_run1
                        val_gain = 'gain_run' + str(run_id + 1) # gain_run1
                        val_loss = 'loss_run' + str(run_id + 1) # loss_run1
                        val_answer = 'answers_run' + str(run_id + 1)
                        onset[val].append(float(info[0])) # onsets for trial_run1
                        duration[val].append(float(4))
                        weights_gain[val_gain].append(float(info[2])) # weights gain for trial_run1
                        weights_loss[val_loss].append(float(info[3])) # weights loss for trial_run1
                        if 'accept' in str(info[5]):
                            answers[val_answer].append(1)
                        else:
                            answers[val_answer].append(0)

        # Bunching is done per run, i.e. trial_run1, trial_run2, etc.
        # But names must not have '_run1' etc because we concatenate runs
        subject_info = []
        for run_id in range(len(runs)):

            conditions = [c + '_run' + str(run_id + 1) for c in condition_names]
            gain = 'gain_run' + str(run_id + 1)
            loss = 'loss_run' + str(run_id + 1)
            answer = 'answers_run' + str(run_id + 1)

            subject_info.insert(
                run_id,
                Bunch(
                    conditions = condition_names,
                    onsets = [onset[c] for c in conditions],
                    durations = [duration[c] for c in conditions],
                    amplitudes = None,
                    tmod = None,
                    pmod = [Bunch(
                        name = [gain, loss, answer],
                        poly=[1, 1, 1],
                        param=[
                            weights_gain[gain],
                            weights_loss[loss],
                            answers[answer]])
                    ],
                    regressor_names = None,
                    regressors=None)
                )

        return subject_info

    def get_contrasts():
        """
        Create the list of tuples that represents contrasts.
        Each contrast is in the form :
        (Name,Stat,[list of condition names],[weights on those conditions])
        """
        # Lists of condition names
        gain = []
        loss = []
        for run_id in range(4):
            run_id += 1
            gain.append(f'gamble_run{run_id}xgain_run{run_id}^1')
            loss.append(f'gamble_run{run_id}xloss_run{run_id}^1')

        # Return contrast list
        return [
            ('pos_gain', 'T', gain, [1, 1, 1, 1]),
            ('pos_loss', 'T', loss, [1, 1, 1, 1]),
            ('neg_gain', 'T', gain, [-1, -1, -1, -1]),
            ('neg_loss', 'T', loss, [-1, -1, -1, -1])
            ]

    def get_subject_level_analysis(self):
        """
        Create the subject level analysis workflow.

        Returns:
            - l1_analysis : nipype.WorkFlow
        """
        # Infosource Node - To iterate on subjects
        infosource = Node(IdentityInterface(
            fields = ['subject_id']),
            name = 'infosource')
        infosource.iterables = [('subject_id', self.subject_list)]

        # Templates to select files node
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
            'event' : join(self.directories.dataset_dir,
                'sub-{subject_id}', 'func', 'sub-{subject_id}_task-MGT_run-*_events.tsv')
        }

        # SelectFiles node - to select necessary files
        selectfiles = Node(SelectFiles(
            templates, base_directory = self.directories.results_dir),
            name = 'selectfiles')

        # DataSink Node - store the wanted results in the wanted repository
        datasink = Node(DataSink(
            base_directory = self.directories.output_dir),
            name = 'datasink')

        # Get Subject Info - get subject specific condition information
        subject_info = Node(Function(
            function = self.get_subject_info),
            input_names = ['event_files', 'runs'],
            output_names = ['subject_info'],
            name = 'subject_info')
        subject_info.inputs.runs = self.run_list

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
        specify_model = Node(SpecifySPMModel(
            concatenate_runs = False, input_units = 'secs', output_units = 'secs',
            time_repetition = self.tr, high_pass_filter_cutoff = 128),
            name='specify_model')

        # Level1Design - Generates an SPM design matrix
        l1_design = Node(Level1Design(
            bases = {'hrf': {'derivs': [1, 1]}}, timing_units = 'secs',
            interscan_interval = self.tr),
            name='l1_design')

        # EstimateModel - estimate the parameters of the model
        l1_estimate = Node(EstimateModel(
            estimation_method = {'Classical': 1}),
            name = 'l1_estimate')

        # Node contrasts to get contrasts
        contrasts = Node(Function(
            function = self.get_contrasts,
            input_names = ['subject_id'],
            output_names = ['contrasts']),
            name = 'contrasts')

        # EstimateContrast - estimates contrasts
        contrast_estimate = Node(EstimateContrast(),
            name = 'contrast_estimate')

        # Create l1 analysis workflow and connect its nodes
        l1_analysis = Workflow(base_dir = self.directories.working_dir, name = 'l1_analysis')

        l1_analysis.connect([(infosource, selectfiles, [('subject_id', 'subject_id')]),
                            (infosource, contrasts, [('subject_id', 'subject_id')]),
                            (infosource, parameters, [('subject_id', 'subject_id')]),
                            (subject_info, specify_model, [('subject_info', 'subject_info')]),
                            (contrasts, contrast_estimate, [('contrasts', 'contrasts')]),
                            (selectfiles, subject_info, [('event', 'event_files')]),
                            (selectfiles, parameters, [
                                ('motion_correction', 'motion_corrected_files'),
                                ('param', 'parameters_files'),
                                ('wc2', 'wc2_file')]),
                            (selectfiles, specify_model, [('func', 'functional_runs')]),
                            (parameters, specify_model, [
                                ('new_parameters_files', 'realignment_parameters')]),
                            (specify_model, l1_design, [('session_info', 'session_info')]),
                            (l1_design, l1_estimate, [('spm_mat_file', 'spm_mat_file')]),
                            (l1_estimate, contrast_estimate, [
                                ('spm_mat_file', 'spm_mat_file'),
                                ('beta_images', 'beta_images'),
                                ('residual_image', 'residual_image')]),
                            (contrast_estimate, datasink, [
                                ('con_images', 'l1_analysis.@con_images'),
                                ('spmT_images', 'l1_analysis.@spmT_images'),
                                ('spm_mat_file', 'l1_analysis.@spm_mat_file')])
                            ])

        return l1_analysis

    def get_subject_level_outputs(self):
        """ Return the names of the files the subject level analysis is supposed to generate. """

        # Contrat maps
        templates = [join(
            self.directories.output_dir,
            'l1_analysis', '_subject_id_{subject_id}', f'con_{contrast_id}.nii')\
            for contrast_id in self.contrast_list]

        # SPM.mat file
        templates += [join(
            self.directories.output_dir,
            'l1_analysis', '_subject_id_{subject_id}', 'SPM.mat')]

        # spmT maps
        templates += [join(
            self.directories.output_dir,
            'l1_analysis', '_subject_id_{subject_id}', f'spmT_{contrast_id}.nii')\
            for contrast_id in self.contrast_list]

        # Format with subject_ids
        return_list = []
        for template in templates:
            return_list += [template.format(subject_id = s) for s in self.subject_list]

        return return_list

    def get_subset_contrasts(file_list, subject_list, participants_file):
        """
        Parameters :
        - file_list : original file list selected by selectfiles node
        - subject_list : list of subject IDs that are in the wanted group for the analysis
        - participants_file: str, file containing participants characteristics

        This function return the file list containing only the files belonging to subject
        in the wanted group.
        """
        equal_indifference_id = []
        equal_range_id = []
        equal_indifference_files = []
        equal_range_files = []

        with open(participants_file, 'rt') as file:
            next(file)  # skip the header
            for line in file:
                info = line.strip().split()

                if info[0][-3:] in subject_list and info[1] == 'equalIndifference':
                    equal_indifference_id.append(info[0][-3:])
                elif info[0][-3:] in subject_list and info[1] == 'equalRange':
                    equal_range_id.append(info[0][-3:])

        for file in file_list:
            sub_id = file.split('/')
            if sub_id[-2][-3:] in equal_indifference_id:
                equal_indifference_files.append(file)
            elif sub_id[-2][-3:] in equal_range_id:
                equal_range_files.append(file)

        return equal_indifference_id, equal_range_id, equal_indifference_files, equal_range_files

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
            - l2_analysis: Nipype WorkFlow
        """
        # Compute the number of participants used to do the analysis
        nb_subjects = len(self.subject_list)

        # Infosource - a function free node to iterate over the list of subject names
        infosource_groupanalysis = Node(
            IdentityInterface(
                fields = ['contrast_id', 'subjects']),
                name = 'infosource_groupanalysis')
        infosource_groupanalysis.iterables = [('contrast_id', self.contrast_list)]

        # SelectFiles
        templates = {
            'contrast' : join(self.directories.output_dir,
                'l1_analysis', '_subject_id_*', 'con_{contrast_id}.nii'),
            'participants' : join(self.directories.dataset_dir, 'participants.tsv')
        }

        selectfiles_groupanalysis = Node(SelectFiles(
            templates, base_directory = self.directories.results_dir, force_list = True),
            name = 'selectfiles_groupanalysis')

        # Datasink - save important files
        datasink_groupanalysis = Node(DataSink(
            base_directory = self.directories.output_dir
            ),
            name = 'datasink_groupanalysis')

        # Node to select subset of contrasts
        sub_contrasts = Node(Function(
            function = self.get_subset_contrasts,
            input_names = ['file_list', 'subject_list', 'participants_file'],
            output_names = [
                'equalIndifference_id',
                'equalRange_id',
                'equalIndifference_files',
                'equalRange_files']),
            name = 'sub_contrasts')
        sub_contrasts.inputs.subject_list = self.subject_list

        # Estimate model
        estimate_model = Node(EstimateModel(
            estimation_method = {'Classical':1}),
            name = 'estimate_model')

        # Estimate contrasts
        estimate_contrast = Node(EstimateContrast(
            group_contrast = True),
            name = 'estimate_contrast')

        # Create thresholded maps
        threshold = MapNode(Threshold(
            use_fwe_correction = False, height_threshold = 0.01,
            extent_fdr_p_threshold = 0.05, use_topo_fdr = False,
            force_activation = True),
            name = 'threshold', iterfield = ['stat_image', 'contrast_index'])

        l2_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = f'l2_analysis_{method}_nsub_{nb_subjects}')
        l2_analysis.connect([
            (infosource_groupanalysis, selectfiles_groupanalysis, [
                ('contrast_id', 'contrast_id')]),
            (selectfiles_groupanalysis, sub_contrasts, [
                ('contrast', 'file_list'),
                ('participants', 'participants_file')]),
            (estimate_model, estimate_contrast, [
                ('spm_mat_file', 'spm_mat_file'),
                ('residual_image', 'residual_image'),
                ('beta_images', 'beta_images')]),
            (estimate_contrast, threshold, [
                ('spm_mat_file', 'spm_mat_file'),
                ('spmT_images', 'stat_image')]),
            (threshold, datasink_groupanalysis, [
                ('thresholded_map', f'l2_analysis_{method}_nsub_{nb_subjects}.@thresh')]),
            (estimate_model, datasink_groupanalysis, [
                ('mask_image', f'l2_analysis_{method}_nsub_{nb_subjects}.@mask')]),
            (estimate_contrast, datasink_groupanalysis, [
                ('spm_mat_file', f'l2_analysis_{method}_nsub_{nb_subjects}.@spm_mat'),
                ('spmT_images', f'l2_analysis_{method}_nsub_{nb_subjects}.@T'),
                ('con_images', f'l2_analysis_{method}_nsub_{nb_subjects}.@con')])])

        if method in ('equalRange', 'equalIndifference'):
            contrasts = [('Group', 'T', ['mean'], [1]), ('Group', 'T', ['mean'], [-1])]

            threshold.inputs.contrast_index = [1, 2]
            threshold.synchronize = True

            ## Specify design matrix
            one_sample_t_test_design = Node(OneSampleTTestDesign(),
                name = 'one_sample_t_test_design')

            l2_analysis.connect([
                (sub_contrasts, one_sample_t_test_design, [(f'{method}_files', 'in_files')]),
                (one_sample_t_test_design, estimate_model, [('spm_mat_file', 'spm_mat_file')])])

        elif method == 'groupComp':
            contrasts = [
            ('Eq range vs Eq indiff in loss', 'T', ['Group_{1}', 'Group_{2}'], [1, -1])]

            threshold.inputs.contrast_index = [1]
            threshold.synchronize = True

            # Node for the design matrix
            two_sample_t_test_design = Node(TwoSampleTTestDesign(),
                name = 'two_sample_t_test_design')

            l2_analysis.connect([
                (sub_contrasts, two_sample_t_test_design, [
                    ('equalRange_files', 'group1_files'),
                    ('equalIndifference_files', 'group2_files')]),
                (two_sample_t_test_design, estimate_model, [
                    ('spm_mat_file', 'spm_mat_file')])
                ])

        estimate_contrast.inputs.contrasts = contrasts

        return l2_analysis

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
            'l2_analysis_{method}_nsub_{nb_subjects}',
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
            'l2_analysis_{method}_nsub_{nb_subjects}',
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
            join(f'l2_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'l2_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001', 'spmT_0001.nii'),
            # Hypothesis 2
            join(f'l2_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'l2_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001', 'spmT_0001.nii'),
            # Hypothesis 3
            join(f'l2_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'l2_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001', 'spmT_0001.nii'),
            # Hypothesis 4
            join(f'l2_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'l2_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001', 'spmT_0001.nii'),
            # Hypothesis 5
            join(f'l2_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold0', 'spmT_0002_thr.nii'),
            join(f'l2_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0002.nii'),
            # Hypothesis 6
            join(f'l2_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold1', 'spmT_0002_thr.nii'),
            join(f'l2_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0002.nii'),
            # Hypothesis 7
            join(f'l2_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'l2_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0001.nii'),
            # Hypothesis 8
            join(f'l2_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold1', 'spmT_0001_thr.nii'),
            join(f'l2_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0001.nii'),
            # Hypothesis 9
            join(f'l2_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'l2_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0001.nii')
        ]
        return [join(self.directories.output_dir, f) for f in files]
