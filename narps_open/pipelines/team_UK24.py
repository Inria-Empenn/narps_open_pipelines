#!/usr/bin/python
# coding: utf-8

""" Write the work of NARPS' team UK24 using Nipype """

from os.path import join
from itertools import product

from nipype import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.utility.base import Select, Merge, Split
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.spm import (
    Coregister, Segment, Reslice, Realign,
    Smooth, Level1Design, OneSampleTTestDesign, TwoSampleTTestDesign,
    EstimateModel, EstimateContrast, Threshold
    )
from nipype.algorithms.confounds import FramewiseDisplacement
from nipype.algorithms.modelgen import SpecifySPMModel
from nipype.algorithms.misc import Gunzip, SimpleThreshold

from narps_open.pipelines import Pipeline
from narps_open.data.task import TaskInformation
from narps_open.data.participants import get_group
from narps_open.core.common import (
    remove_file, list_intersection, elements_in_string, clean_list
    )

class PipelineTeamUK24(Pipeline):
    """ A class that defines the pipeline of team UK24. """

    def __init__(self):
        super().__init__()
        self.fwhm = 4.0
        self.team_id = 'UK24'
        self.contrast_list = ['0001', '0002']
        condition_names = [
            'gain', 'gainxgain_value^1', 'gainxgain_rt^1',
            'loss', 'lossxloss_value^1', 'lossxloss_rt^1',
            'no_gain_no_loss', 'no_gain_no_lossxno_gain_no_loss_rt^1'
            ]
        self.subject_level_contrasts = [
            ['gain_param', 'T', condition_names, [0, 1, 1, 0, 0, 0, 0, 0]],
            ['loss_param', 'T', condition_names, [0, 0, 0, 0, 1, 1, 0, 0]]
            ]

    def get_average_values(
        in_file: str,
        mask: str,
        subject_id: str,
        run_id: str,
        out_file_suffix: str
        ) -> str:
        """
        Compute the average value of each frame of a 4D image, while applying a mask.
        Write the values in a text file.

        Parameters:
            - in_file: str, path to the input file (4D)
            - mask: str, path to the mask file (3D) Zeroes in mask are considered as 0,
                other values are 1
            - subject_id: str, related subject id
            - run_id: str, related run id
            - out_file_suffix: str, string to append at the end of the file name

        Returns:
            - preprocessing : nipype.WorkFlow
        """
        from os import makedirs
        from os.path import join, abspath
        from nibabel import load
        from numpy import mean

        # Load input images
        mask_image = load(mask)
        in_file_image = load(in_file)

        # Create mask
        mask_data = mask_image.get_fdata() > 0.0
        mask_data = mask_data.astype(int)

        # Loop through time points
        average_values = []
        for index in range(in_file_image.shape[3]):
            average_values.append(mean(in_file_image.slicer[..., index].get_fdata() * mask_data))

        # Write confounds to a file
        out_file_name = abspath(f'sub-{subject_id}_run-{run_id}_' + out_file_suffix)

        # Write output file
        with open(out_file_name, 'w', encoding = 'utf-8') as writer:
            for value in average_values:
                writer.write(f'{value}\n')

        return out_file_name

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
            'anat' : join('sub-{subject_id}', 'anat', 'sub-{subject_id}_T1w.nii.gz'),
            'func' : join('sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-*_bold.nii.gz'),
            'sbref' : join('sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-*_sbref.nii.gz'),
        }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir
        preprocessing.connect(information_source, 'subject_id', select_files, 'subject_id')

        # GUNZIP - gunzip files because SPM do not use .nii.gz files
        gunzip_anat = Node(Gunzip(), name = 'gunzip_anat')
        gunzip_func = MapNode(Gunzip(), name = 'gunzip_func', iterfield = ['in_file'])
        gunzip_sbref = MapNode(Gunzip(), name = 'gunzip_sbref', iterfield = ['in_file'])
        preprocessing.connect(select_files, 'anat', gunzip_anat, 'in_file')
        preprocessing.connect(select_files, 'func', gunzip_func, 'in_file')
        preprocessing.connect(select_files, 'sbref', gunzip_sbref, 'in_file')

        # SELECT - Select the first sbref file from the selected (and gunzipped) files.
        select_first_sbref = Node(Select(), name = 'select_first_sbref')
        select_first_sbref.inputs.index = [0]
        preprocessing.connect(gunzip_sbref, 'out_file', select_first_sbref, 'inlist')

        # COREGISTER - Coregistration of the structural T1w image to the reference functional image
        #   (defined as the single volume 'sbref' image acquired before the first functional run).
        coregistration = Node(Coregister(), name = 'coregistration')
        coregistration.inputs.cost_function = 'nmi'
        preprocessing.connect(gunzip_anat, 'out_file', coregistration, 'source')
        preprocessing.connect(select_first_sbref, 'out', coregistration, 'target')

        # SEGMENT - Segmentation of the coregistered T1w image into grey matter, white matter and
        #   cerebrospinal fluid tissue maps.
        segmentation = Node(Segment(), name = 'segmentation')
        segmentation.inputs.bias_fwhm = 60 #mm
        #segmentation.inputs.bias_regularization = (0 or 1e-05 or 0.0001 or 0.001 or 0.01 or 0.1 or 1 or 10) "light regularization"
        segmentation.inputs.csf_output_type = [False,False,True] # Output maps in native space only
        segmentation.inputs.gm_output_type = [False,False,True] # Output maps in native space only
        segmentation.inputs.wm_output_type = [False,False,True] # Output maps in native space only
        preprocessing.connect(coregistration, 'coregistered_source', segmentation, 'data')

        # MERGE - Merge files for the reslicing node into one input.
        merge_before_reslicing = Node(Merge(4), name = 'merge_before_reslicing')
        preprocessing.connect(coregistration, 'coregistered_source', merge_before_reslicing, 'in1')
        preprocessing.connect(segmentation, 'native_csf_image', merge_before_reslicing, 'in2')
        preprocessing.connect(segmentation, 'native_wm_image', merge_before_reslicing, 'in3')
        preprocessing.connect(segmentation, 'native_gm_image', merge_before_reslicing, 'in4')

        # RESLICE - Reslicing of coregistered T1w image and segmentations to the same voxel space
        #   of the reference (sbref) image.
        reslicing = MapNode(Reslice(), name = 'reslicing', iterfield = 'in_file')
        preprocessing.connect(merge_before_reslicing, 'out', reslicing, 'in_file')
        preprocessing.connect(select_first_sbref, 'out', reslicing, 'space_defining')

        # REALIGN - Realigning all 4 sbref files images to the one acquired before the first run
        #   Realigns to the first image of the list by default. TODO : check
        #   Note : gunzip_sbref.out_file is a list of files
        realign_sbref = Node(Realign(), name = 'realign_sbref')
        realign_sbref.inputs.register_to_mean = False
        preprocessing.connect(gunzip_sbref, 'out_file', realign_sbref, 'in_files')

        # REALIGN - Realigning all volumes in a particular run to the first image in that run.
        #   Realigns to the first image of the run by default.
        #   Note : gunzip_func.out_file is a list of files, but we want to realign run by run,
        #   therefore, passing one func file at a time.
        realign_func = MapNode(Realign(), name = 'realign_func', iterfield = 'in_files')
        realign_func.inputs.register_to_mean = False
        preprocessing.connect(gunzip_func, 'out_file', realign_func, 'in_files')

        # SMOOTH - Spatial smoothing of fMRI data.
        #   Note : realign_func.realigned_files will be a list(list(files)) :
        #   we need a MapNode to process it.
        smoothing = MapNode(Smooth(), name = 'smoothing', iterfield = 'in_files')
        smoothing.inputs.fwhm = [self.fwhm] * 3
        preprocessing.connect(realign_func, 'realigned_files', smoothing, 'in_files')

        # SELECT - Select the segmentation probability maps.
        select_maps = Node(Select(), name = 'select_maps')
        select_maps.inputs.index = [1, 2] # csf and wm only
        preprocessing.connect(reslicing, 'out_file', select_maps, 'inlist')

        # SIMPLE THRESHOLD - Apply a threshold to proability maps
        threshold = Node(SimpleThreshold(), name = 'threshold')
        threshold.inputs.threshold = 0.5
        preprocessing.connect(select_maps, 'out', threshold, 'volumes')

        # FRAMEWISE DISPLACEMENT - Compute framewise displacement from realignment parameters.
        displacement = MapNode(
            FramewiseDisplacement(),
            name = 'displacement',
            iterfield = 'in_file'
            )
        displacement.inputs.parameter_source = 'SPM'
        preprocessing.connect(realign_func, 'realignment_parameters', displacement, 'in_file')

        # SPLIT - Select the WM and CSF masks.
        split_masks = Node(Split(), name = 'split_masks')
        split_masks.inputs.splits = [1, 1]
        split_masks.inputs.squeeze = True # Unfold one-element splits removing the list.
        preprocessing.connect(threshold, 'thresholded_volumes', split_masks, 'inlist')

        # FUNCTION - Create text files with average values in WM at each timepoint
        #   to be later used as regressor.
        average_values_wm = MapNode(
            Function(
                function = self.get_average_values,
                input_names = ['in_file', 'mask', 'subject_id', 'run_id', 'out_file_suffix'],
                output_names = ['out_file']
            ),
            iterfield = ['run_id', 'in_file'],
            name = 'average_values_wm')
        average_values_wm.inputs.run_id = self.run_list
        average_values_wm.inputs.out_file_suffix = join('reg_wm.txt')
        preprocessing.connect(information_source, 'subject_id', average_values_wm, 'subject_id')
        preprocessing.connect(smoothing, 'smoothed_files', average_values_wm, 'in_file')
        preprocessing.connect(split_masks, 'out2', average_values_wm, 'mask')

        # FUNCTION - Create text files with average values in CSF at each timepoint
        #   to be later used as regressor.
        average_values_csf = MapNode(
            Function(
                function = self.get_average_values,
                input_names = ['in_file', 'mask', 'subject_id', 'run_id', 'out_file_suffix'],
                output_names = ['out_file']
            ),
            iterfield = ['run_id', 'in_file'],
            name = 'average_values_csf')
        average_values_csf.inputs.run_id = self.run_list
        average_values_csf.inputs.out_file_suffix = join('reg_csf.txt')
        preprocessing.connect(information_source, 'subject_id', average_values_csf, 'subject_id')
        preprocessing.connect(smoothing, 'smoothed_files', average_values_csf, 'in_file')
        preprocessing.connect(split_masks, 'out1', average_values_csf, 'mask')

        # DATA SINK - store the wanted results in the wanted repository
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir
        preprocessing.connect(smoothing, 'smoothed_files', data_sink, 'preprocessing.@smoothed')
        preprocessing.connect(segmentation, 'transformation_mat', data_sink, 'preprocessing.@tf_to_mni')
        preprocessing.connect(displacement, 'out_file', data_sink, 'preprocessing.@reg_scrubbing')
        preprocessing.connect(
            realign_func, 'realignment_parameters',
            data_sink, 'preprocessing.@realignment_parameters')
        preprocessing.connect(average_values_csf, 'out_file', data_sink, 'preprocessing.@reg_csf')
        preprocessing.connect(average_values_wm, 'out_file', data_sink, 'preprocessing.@reg_wm')

        return preprocessing

    def get_preprocessing_outputs(self):
        """ Return the names of the files the preprocessing is supposed to generate. """

        output_dir = join(self.directories.output_dir, 'preprocessing', '_subject_id_{subject_id}')

        # WM and CSF average values
        templates = [join(output_dir, f'_average_values_wm{index}',
            'sub-{subject_id}'+f'_run-{run_id}_reg_wm.txt')\
            for index, run_id in zip(range(len(self.run_list)), self.run_list)]
        templates += [join(output_dir, f'_average_values_csf{index}',
            'sub-{subject_id}'+f'_run-{run_id}_reg_csf.txt')\
            for index, run_id in zip(range(len(self.run_list)), self.run_list)]

        # Framewise displacement values
        templates += [join(output_dir, f'_displacement{index}',
            'fd_power_2012.txt')\
            for index in range(len(self.run_list))]

        # Realignement parameters
        templates += [join(output_dir, f'_realign_func{index}',
            'rp_sub-{subject_id}'+f'_task-MGT_run-{run_id}_bold.txt')\
            for index, run_id in zip(range(len(self.run_list)), self.run_list)]

        # Smoothing outputs
        templates += [join(output_dir, f'_smoothing{index}',
            'srsub-{subject_id}'+f'_task-MGT_run-{run_id}_bold.nii')\
            for index, run_id in zip(range(len(self.run_list)), self.run_list)]


        # Transformation matrix to MNI
        templates.append(join(output_dir, 'rsub-{subject_id}_T1w_seg_sn.mat'))

        # Format with subject_ids
        return_list = []
        for template in templates:
            return_list += [template.format(subject_id = s) for s in self.subject_list]

        return return_list

    def get_run_level_analysis(self):
        """ No run level analysis has been done by team UK24 """
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
        from nipype.algorithms.modelgen import orth

        # Init empty lists inside directiries
        onsets = []
        durations = []
        onsets_no_gain_no_loss = []
        durations_no_gain_no_loss = []
        gain_value = []
        gain_rt = []
        loss_value = []
        loss_rt = []
        no_gain_no_loss_rt = []

        with open(event_file, 'rt') as file:
            next(file)  # skip the header

            for line in file:
                info = line.strip().split()

                if 'accept' in info[5]:
                    onsets.append(float(info[0]))
                    durations.append(float(info[1]))
                    gain_value.append(float(info[2]))
                    loss_value.append(float(info[3]))
                    gain_rt.append(float(info[4]))
                    loss_rt.append(float(info[4]))
                else:
                    onsets_no_gain_no_loss.append(float(info[0]))
                    durations_no_gain_no_loss.append(float(info[1]))
                    no_gain_no_loss_rt.append(float(info[4]))

        # NOTE : SPM automatically mean-centers regressors
        #gain_value = list(gain_value - mean(gain_value))
        #loss_value = list(loss_value - mean(loss_value))
        #gain_rt = list(gain_rt - mean(gain_rt))
        #loss_rt = list(loss_rt - mean(loss_rt))
        #no_gain_no_loss_rt = list(no_gain_no_loss_rt - mean(no_gain_no_loss_rt))

        # Othogonalize
        #gain_rt = orth(gain_value, gain_rt)
        #loss_rt = orth(loss_value, loss_rt)

        # Fill Bunch
        return Bunch(
                conditions = ['gain', 'loss', 'no_gain_no_loss'],
                onsets = [onsets, onsets, onsets_no_gain_no_loss],
                durations = [durations, durations, durations_no_gain_no_loss],
                amplitudes = None,
                tmod = None,
                pmod = [
                    Bunch(
                        name = ['gain_value', 'gain_rt'],
                        poly = [1, 1],
                        param = [gain_value, gain_rt]
                    ),
                    Bunch(
                        name = ['loss_value', 'loss_rt'],
                        poly = [1, 1],
                        param = [loss_value, loss_rt]
                    ),
                    Bunch(
                        name = ['no_gain_no_loss_rt'],
                        poly = [1],
                        param = [no_gain_no_loss_rt]
                    )
                ],
                regressor_names = None,
                regressors = None
            )

    def get_confounds_file(
        framewise_displacement_file: str,
        wm_average_file: str,
        csf_average_file: str,
        realignement_parameters: str,
        subject_id: str,
        run_id: str) -> str:
        """
        Create a tsv file with only desired confounds per subject per run.

        Parameters :
        - framewise_displacement_file: str, path to the framewise displacement file
        - wm_average_file: str, path to the WM average values file
        - csf_average_file: str, path to the CSF average values file
        - realignement_parameters : path to the realignment parameters file
        - subject_id : related subject id
        - run_id : related run id

        Return :
        - confounds_file : path to new file containing only desired confounds
        """
        from os.path import abspath

        from pandas import DataFrame, read_csv
        from numpy import array, transpose, concatenate, insert

        # Get the dataframe containing the 6 head motion parameter regressors
        realign_data_frame = array(read_csv(realignement_parameters, sep = r'\s+', header = None))

        # Get the dataframes containing the 2 tissue signal regressors
        regressor_csf_data_frame = array(read_csv(csf_average_file, sep = '\t', header = None))
        regressor_wm_data_frame = array(read_csv(wm_average_file, sep = '\t', header = None))

        # Get the dataframe containing framewise displacement
        # and transform it as a scrubbing regressor.
        scrubbing_data_frame = insert(
            array(
            (read_csv(framewise_displacement_file, sep = '\t', header = 0) > 0.5).astype(int)
            ),
            0, 0, axis = 0) # Add a value of 0 at the beginning (first frame)

        # Extract all parameters
        retained_parameters = DataFrame(
            concatenate(
                (realign_data_frame,
                 regressor_csf_data_frame, regressor_wm_data_frame, scrubbing_data_frame),
                axis = 1)
        )

        # Write confounds to a file
        confounds_file = abspath(f'confounds_file_sub-{subject_id}_run-{run_id}.tsv')
        with open(confounds_file, 'w', encoding = 'utf-8') as writer:
            writer.write(retained_parameters.to_csv(
                sep = '\t', index = False, header = False, na_rep = '0.0'))

        return confounds_file

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
            'framewise_displacement_files' : join(self.directories.output_dir, 'preprocessing',
                '_subject_id_{subject_id}', '_displacement*',
                'fd_power_2012.txt'),
            'wm_average_files' : join(self.directories.output_dir, 'preprocessing',
                '_subject_id_{subject_id}', '_average_values_wm*',
                'sub-{subject_id}_run-*_reg_wm.txt'),
            'csf_average_files' : join(self.directories.output_dir, 'preprocessing',
                '_subject_id_{subject_id}', '_average_values_csf*',
                'sub-{subject_id}_run-*_reg_csf.txt'),
            'realignement_parameters' : join(self.directories.output_dir, 'preprocessing',
                '_subject_id_{subject_id}', '_realign_func*',
                'rp_sub-{subject_id}_task-MGT_run-*_bold.txt'),
            'func' : join(self.directories.output_dir, 'preprocessing', '_subject_id_{subject_id}',
                '_smoothing*', 'srsub-{subject_id}_task-MGT_run-*_bold.nii'),
            'event' : join('sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-*_events.tsv'),
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
                input_names = ['framewise_displacement_file', 'wm_average_file',
                    'csf_average_file', 'realignement_parameters', 'subject_id', 'run_id'],
                output_names = ['confounds_file']
            ),
            name = 'confounds',
            iterfield = ['framewise_displacement_file', 'wm_average_file',
                'csf_average_file', 'realignement_parameters', 'run_id'])
        confounds.inputs.run_id = self.run_list
        subject_level.connect(information_source, 'subject_id', confounds, 'subject_id')
        subject_level.connect(
            select_files, 'framewise_displacement_files', confounds, 'framewise_displacement_file')
        subject_level.connect(select_files, 'wm_average_files', confounds, 'wm_average_file')
        subject_level.connect(select_files, 'csf_average_files', confounds, 'csf_average_file')
        subject_level.connect(
            select_files, 'realignement_parameters', confounds, 'realignement_parameters')

        # SPECIFY MODEL - generates SPM-specific Model
        specify_model = Node(SpecifySPMModel(), name = 'specify_model')
        #specify_model.inputs.concatenate_runs = True # TODO : actually concatenate runs ????
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
