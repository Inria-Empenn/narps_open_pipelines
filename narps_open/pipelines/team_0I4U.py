#!/usr/bin/python
# coding: utf-8

""" Write the work of NARPS team 0I4U using Nipype """
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

class PipelineTeam0I4U(Pipeline):
    """ A class that defines the pipeline of team 0I4U. """

    def __init__(self):
        super().__init__()
        self.fwhm = 5.0
        self.team_id = '0I4U'
        self.contrast_list = ['0001', '0002']

        # Define contrasts
        gain_conditions = [f'trial_run{r}xgain_run{r}^1' for r in range(1,len(self.run_list) + 1)]
        loss_conditions = [f'trial_run{r}xloss_run{r}^1' for r in range(1,len(self.run_list) + 1)]
        self.subject_level_contrasts = [
            ('gain', 'T', gain_conditions, [1] * len(self.run_list)),
            ('loss', 'T', loss_conditions, [1] * len(self.run_list))
            ]

    def get_preprocessing(self):
        """
        Return the preprocessing workflow.

        Returns: a nipype.WorkFlow 
        """

        # Initialize workflow
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

        # SELECT FILES - select subject files
        templates = {
            'anat' : join('sub-{subject_id}', 'anat', 'sub-{subject_id}_T1w.nii.gz'),
            'magnitude' : join('sub-{subject_id}', 'fmap', 'sub-{subject_id}_magnitude1.nii.gz'),
            'phasediff' : join('sub-{subject_id}', 'fmap', 'sub-{subject_id}_phasediff.nii.gz')
        }
        select_subject_files = Node(SelectFiles(templates), name = 'select_subject_files')
        select_subject_files.inputs.base_directory = self.directories.dataset_dir
        preprocessing.connect(
            information_source_subject, 'subject_id', select_subject_files, 'subject_id')

        # SELECT FILES - select run files
        template = {
            'func' : join('sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold.nii.gz')
        }
        select_run_files = Node(SelectFiles(template), name = 'select_run_files')
        select_run_files.inputs.base_directory = self.directories.dataset_dir
        preprocessing.connect(
            information_source_runs, 'subject_id', select_run_files, 'subject_id')
        preprocessing.connect(information_source_runs, 'run_id', select_run_files, 'run_id')

        # GUNZIP NODE - SPM do not use .nii.gz files
        gunzip_anat = Node(Gunzip(), name = 'gunzip_anat')
        gunzip_func = Node(Gunzip(), name = 'gunzip_func')
        gunzip_magnitude = Node(Gunzip(), name = 'gunzip_magnitude')
        gunzip_phasediff = Node(Gunzip(), name = 'gunzip_phasediff')
        preprocessing.connect(select_subject_files, 'anat', gunzip_anat, 'in_file')
        preprocessing.connect(select_subject_files, 'magnitude', gunzip_magnitude, 'in_file')
        preprocessing.connect(select_subject_files, 'phasediff', gunzip_phasediff, 'in_file')
        preprocessing.connect(select_run_files, 'func', gunzip_func, 'in_file')

        # FIELDMAP - create field map from phasediff and magnitude file
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
        preprocessing.connect(gunzip_magnitude, 'out_file', fieldmap, 'magnitude_file')
        preprocessing.connect(gunzip_phasediff, 'out_file', fieldmap, 'phase_file')
        preprocessing.connect(gunzip_func, 'out_file', fieldmap, 'epi_file')

        # REALIGN UNWARP - motion correction
        motion_correction = Node(RealignUnwarp(), name = 'motion_correction')
        motion_correction.inputs.interp = 4
        motion_correction.inputs.register_to_mean = False
        preprocessing.connect(fieldmap, 'vdm', motion_correction, 'phase_map')
        preprocessing.connect(gunzip_func, 'out_file', motion_correction, 'in_files')

        # EXTRACTROI - extracting the first image of func
        extract_first_image = Node(ExtractROI(), name = 'extract_first_image')
        extract_first_image.inputs.t_min = 1
        extract_first_image.inputs.t_size = 1
        extract_first_image.inputs.output_type='NIFTI'
        preprocessing.connect(
            motion_correction, 'realigned_unwarped_files', extract_first_image, 'in_file')

        # COREGISTER - Co-registration in SPM12 using default parameters.
        coregistration = Node(Coregister(), name = 'coregistration')
        coregistration.inputs.jobtype = 'estimate'
        coregistration.inputs.write_mask = False
        preprocessing.connect(gunzip_anat, 'out_file', coregistration, 'target')
        preprocessing.connect(extract_first_image, 'roi_file', coregistration, 'source')
        preprocessing.connect(
            motion_correction, 'realigned_unwarped_files', coregister, 'apply_to_files')

        # Get SPM Tissue Probability Maps file
        spm_tissues_file = join(SPMInfo.getinfo()['path'], 'tpm', 'TPM.nii')

        # NEW SEGMENT - Unified segmentation using tissue probability maps in SPM12.
        segmentation = Node(NewSegment(), name = 'segmentation')
        segmentation.inputs.write_deformation_fields = [False, True]
        segmentation.inputs.tissues = [
            [(spm_tissues_file, 1), 1, (True, False), (True, False)],
            [(spm_tissues_file, 2), 1, (True, False), (True, False)],
            [(spm_tissues_file, 3), 2, (True, False), (True, False)],
            [(spm_tissues_file, 4), 3, (True, False), (True, False)],
            [(spm_tissues_file, 5), 4, (True, False), (True, False)],
            [(spm_tissues_file, 6), 2, (True, False), (True, False)]
        ]
        preprocessing.connect(gunzip_anat, 'out_file', segmentation, 'channel_files')

        # NORMALIZE12 - Spatial normalization of functional images
        normalize = Node(Normalize12(), name = 'normalize')
        normalize.inputs.write_voxel_sizes = [1.5, 1.5, 1.5]
        normalize.inputs.jobtype = 'write'
        preprocessing.connect(
            segmentation, 'forward_deformation_field', normalize, 'deformation_file')
        preprocessing.connect(coregistration, 'coregistered_files', normalize, 'apply_to_files')

        # MASKING - Mask func using segmented and normalized grey matter mask
        # TODO
        #preprocessing.connect(segmentation, 'normalized_class_images', masking, 'mask')
        #OR preprocessing.connect(segmentation, 'modulated_class_images', masking, 'mask')

        # SMOOTHING - 5 mm fixed FWHM smoothing
        smoothing = Node(Smooth(), name = 'smoothing')
        smoothing.inputs.fwhm = self.fwhm
        preprocessing.connect(normalize, 'normalized_files', smoothing, 'in_files')

        # DATASINK - store the wanted results in the wanted repository
        data_sink = Node(DataSink(), name='data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir
        preprocessing.connect(
            motion_correction,'realignment_parameters', data_sink, 'preprocess.@realign_par')
        preprocessing.connect(smoothing, 'smoothed_files', data_sink, 'preprocessing.@smoothed')

        # Remove large files, if requested
        if Configuration()['pipelines']['remove_unused_data']:

            # Merge Node - Merge file names to be removed after datasink node is performed
            merge_removable_files = Node(Merge(11), name = 'merge_removable_files')
            merge_removable_files.inputs.ravel_inputs = True

            # Function Nodes remove_files - Remove sizeable files once they aren't needed
            remove_after_datasink = MapNode(Function(
                function = remove_parent_directory,
                input_names = ['_', 'file_name'],
                output_names = []
                ), name = 'remove_after_datasink', iterfield = 'file_name')

            # Add connections
            preprocessing.connect([
                (gunzip_anat, merge_removable_files, [('out_file', 'in1')]),
                (gunzip_func, merge_removable_files, [('out_file', 'in2')]),
                (gunzip_phasediff, merge_removable_files, [('out_file', 'in3')]),
                (gunzip_magnitude, merge_removable_files, [('out_file', 'in4')]),
                (fieldmap, merge_removable_files, [('vdm', 'in5')]),
                (motion_correction, merge_removable_files, [('realigned_unwarped_files', 'in6')]),
                (extract_first_image, merge_removable_files, [('roi_file', 'in7')]),
                (segmentation, merge_removable_files, [('forward_deformation_field', 'in8')]),
                (coregistration, merge_removable_files, [('coregistered_files', 'in8')]),
                (normalize, merge_removable_files, [('normalized_files', 'in10')]),
                (smoothing, merge_removable_files, [('smoothed_files', 'in11')]),
                (merge_removable_files, remove_after_datasink, [('out', 'file_name')]),
                (data_sink, remove_after_datasink, [('out_file', '_')])
            ])

        return preprocessing

    def get_preprocessing_outputs(self):
        """ Return the names of the files the preprocessing analysis is supposed to generate. """

        # Smoothed maps
        templates = [join(
            self.directories.output_dir,
            'preprocessing', '_subject_id_{subject_id}', '_run_id_{run_id}',
            'swrrsub-{subject_id}_task-MGT_run-{run_id}_bold.nii')]

        # Motion parameters file
        templates += [join(
            self.directories.output_dir,
            'preprocessing', '_subject_id_{subject_id}', '_run_id_{run_id}',
            'rp_sub-{subject_id}_task-MGT_run-{run_id}_bold.txt')]

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
        Create Bunchs of subject event information for specifySPMModel.

        Event-related design, 4 within subject sessions
        1 Condition: Stimulus presentation, onsets based on tsv file, duration 4 seconds
        2 Parametric modulators: Gain and loss modelled with 1st order polynomial expansion
        1 Condition: button press, onsets based on tsv file, duration 0 seconds

        Parameters :
        - event_file: str, events file for a run of a subject
        - short_run_id: str, an identifier for the run corresponding to the event_file
            must be '1' for the first run, '2' for the second run, etc.

        Returns :
        - subject_info : Bunch corresponding to the event file
        """
        from nipype.interfaces.base import Bunch
        
        onset_trial = []
        duration_trial = []
        weights_gain = []
        weights_loss = []
        onset_button = []
        duration_button = []

        with open(event_file, 'rt') as file:
            next(file)  # skip the header
            
                for line in file:
                    info = line.strip().split()
                    
                    onset_trial.append(float(info[0]))
                    duration_trial.append(4.0)
                    weights_gain.append(float(info[2]))
                    weights_loss.append(float(info[3]))
                    onset_button.append(float(info[0]) + float(info[4]))
                    duration_button.append(0.0)

        # Create bunch
        return Bunch(
            conditions = [f'trial_run{short_run_id}', f'button_run{short_run_id}'],
            onsets = [onset_trial, onset_button],
            durations = [duration_trial, duration_button],
            amplitudes = None,
            tmod = None,
            pmod = [
                Bunch(
                    name = [f'gain_run{short_run_id}', f'loss_run{short_run_id}'],
                    poly = [1, 1],
                    param = [weights_gain, weights_loss]
                )
            ],
            regressor_names = None,
            regressors = None
        )

        return subject_info

    def get_subject_level_analysis(self):
        """ Return a Nipype workflow describing the subject level analysis part of the pipeline """

        # Workflow initialization
        subject_level_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = 'subject_level_analysis'
        )

        # IDENTITY INTERFACE - Allows to iterate on subjects
        information_source = Node(IdentityInterface(fields = ['subject_id']),
            name = 'information_source')
        information_source.iterables = [('subject_id', self.subject_list)]

        # SELECTFILES - to select necessary files
        templates = {
            'func': join(self.directories.output_dir, 'preprocessing',
                '_subject_id_{subject_id}', '_run_id_*',
                'wusub-{subject_id}_task-MGT_run-*_bold.nii',
            ),
            'event': join('sub-{subject_id}', 'func', 'sub-{subject_id}_task-MGT_run-*_events.tsv',
            ),
            'parameters': join(self.directories.output_dir, 'preprocessing',
                '_subject_id_{subject_id}', '_run_id_*',
                'rp_sub-{subject_id}_task-MGT_run-*_bold.txt',
            )
        }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir
        subject_level_analysis.connect(
            information_source, 'subject_id', select_files, 'subject_id')

        # FUNCTION node get_subject_information - get subject specific condition information
        subject_information = MapNode(Function(
                function = self.get_subject_information,
                input_names = ['event_file', 'short_run_id'],
                output_names = ['subject_info']),
            name = 'subject_information', iterfield = ['event_file', 'short_run_id'])
        subject_information.inputs.short_run_id = list(range(1, len(self.run_list) + 1))
        subject_level_analysis.connect(select_files, 'event', subject_information, 'event_file')
        
        # SPECIFY MODEL - Generates SPM-specific Model
        specify_model = Node(SpecifySPMModel(), name='specify_model')
        specify_model.inputs.concatenate_runs = False
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
        model_design.inputs.bases = {'hrf': {'derivs': [1, 0]}}
        model_design.inputs.timing_units = 'secs'
        model_design.inputs.interscan_interval = TaskInformation()['RepetitionTime']
        model_design.inputs.model_serial_correlations = 'AR(1)'
        subject_level_analysis.connect(specify_model, 'session_info', model_design, 'session_info')

        # ESTIMATE MODEL - estimate the parameters of the model
        model_estimate = Node(EstimateModel(), name = 'model_estimate')
        model_estimate.inputs.estimation_method = {'Classical': 1}
        model_estimate.inputs.write_residuals = False
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

        # DataSink Node - store the wanted results in the wanted repository
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir
        subject_level_analysis.connect(
            contrast_estimate, 'con_images', data_sink, 'subject_level_analysis.@con_images')
        subject_level_analysis.connect(
            contrast_estimate, 'spmT_images', data_sink, 'subject_level_analysis.@spmT_images')
        subject_level_analysis.connect(
            contrast_estimate, 'spm_mat_file', data_sink, 'subject_level_analysis.@spm_mat_file')

        return subject_level_analysis

    def get_subset_contrasts(file_list, subject_list, participants_file, method):
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
        equalRange_covar_val = [[], []]
        equalIndifference_covar_val = [[], []]
        """
        covariates (a list of items which are a dictionary with keys which are ‘vector’ or ‘name’ or ‘interaction’ 
        or ‘centering’ and with values which are any value) – Covariate dictionary {vector, name, interaction,
        centering}.
        """

        with open(participants_file, 'rt') as f:
                next(f)  # skip the header
                
                for line in f:
                    info = line.strip().split()
                    
                    if info[0][-3:] in subject_list and info[1] == "equalIndifference":
                        equalIndifference_id.append(info[0][-3:])
                        equalIndifference_covar_val[0].append(float(info[3]))
                        equalIndifference_covar_val[1].append(0 if info[2]=='M' else 1)
                    elif info[0][-3:] in subject_list and info[1] == "equalRange":
                        equalRange_id.append(info[0][-3:])
                        equalRange_covar_val[0].append(float(info[3]))
                        equalRange_covar_val[1].append(0 if info[2]=='M' else 1)
        
        for file in file_list:
            sub_id = file.split('/')
            if sub_id[-2][-3:] in equalIndifference_id:
                equalIndifference_files.append(file)
            elif sub_id[-2][-3:] in equalRange_id:
                equalRange_files.append(file)
                
        equalRange_covar = [dict(vector=equalRange_covar_val[0], name='age', centering=[1]),
                            dict(vector=equalRange_covar_val[1], name='sex', centering=[1])]
        
        equalIndifference_covar = [dict(vector=equalIndifference_covar_val[0], name='age', centering=[1]),
                                   dict(vector=equalIndifference_covar_val[1], name='sex', centering=[1])]
        
        nb_eqRange = len(equalRange_id)
        nb_eqIndifference = len(equalIndifference_id)
        
        global_covar = []
        
        if method == 'groupComp':
            global_covar = [dict(vector=equalRange_covar_val[0], name='eqRange_age'),
                            dict(vector=equalIndifference_covar_val[0], name='eqIndifference_age'),
                            dict(vector=equalRange_covar_val[1], name='eqRange_sex'),
                            dict(vector=equalIndifference_covar_val[1], name='eqIndifference_sex')]

            for i in range(4):
                if i in [0, 2]:
                    for k in range(nb_eqIndifference):
                        global_covar[i]['vector'].append(0)
                elif i in [1, 3]:
                    for k in range(nb_eqRange):
                        global_covar[i]['vector'].insert(0 ,0)
        
                
        return equalIndifference_id, equalRange_id, equalIndifference_files, equalRange_files, equalRange_covar, equalIndifference_covar, global_covar 

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
                fields = ['contrast_id', 'subjects']),
                name = 'information_source')
        information_source.iterables = [('contrast_id', self.contrast_list)]

        # SelectFiles
        contrast_file = opj(result_dir, output_dir, 'l1_analysis', '_subject_id_*', "con_00{contrast_id}.nii")

        participants_file = opj(exp_dir, 'participants.tsv')
        
        mask_file = opj(data_dir, 'NARPS-0I4U', 'hypo1_unthresh.nii.gz')

        templates = {'contrast' : contrast_file, 'participants' : participants_file, 'mask':mask_file}
        
        selectfiles_groupanalysis = Node(SelectFiles(templates, base_directory=result_dir, force_list= True),
                           name="selectfiles_groupanalysis")
        
        # Datasink node : to save important files 
        datasink_groupanalysis = Node(DataSink(base_directory = result_dir, container = output_dir), 
                                      name = 'datasink_groupanalysis')
        
        gunzip_mask = Node(Gunzip(), name='gunzip_mask')
        # Node to select subset of contrasts
        sub_contrasts = Node(Function(input_names = ['file_list', 'subject_list', 'participants_file', 'method'],
                                     output_names = ['equalIndifference_id', 'equalRange_id', 
                                                     'equalIndifference_files', 'equalRange_files',
                                                    'equalRange_covar', 'equalIndifference_covar', 'global_covar'],
                                     function = get_subset_contrasts),
                            name = 'sub_contrasts')
        
        sub_contrasts.inputs.method = method

        ## Estimate model 
        estimate_model = Node(EstimateModel(estimation_method={'Classical':1}), name = "estimate_model")

        ## Estimate contrasts
        estimate_contrast = Node(EstimateContrast(group_contrast=True),
                                 name = "estimate_contrast")

        ## Create thresholded maps 
        threshold = MapNode(Threshold(use_fwe_correction=True,
                                      height_threshold_type='p-value',
                                      force_activation = False), name = "threshold", 
                            iterfield = ["stat_image", "contrast_index"])
        

        l2_analysis = Workflow(base_dir = opj(result_dir, working_dir), name = f"l2_analysis_{method}_nsub_{n_sub}")

        l2_analysis.connect([(infosource_groupanalysis, selectfiles_groupanalysis, [('contrast_id', 'contrast_id')]),
            (infosource_groupanalysis, sub_contrasts, [('subjects', 'subject_list')]),
            (selectfiles_groupanalysis, sub_contrasts, [('contrast', 'file_list'),
                                                        ('participants', 'participants_file')]),
            (selectfiles_groupanalysis, gunzip_mask, [('mask', 'in_file')]),
            (estimate_model, estimate_contrast, [('spm_mat_file', 'spm_mat_file'),
                ('residual_image', 'residual_image'),
                ('beta_images', 'beta_images')]),
            (estimate_contrast, threshold, [('spm_mat_file', 'spm_mat_file'),('spmT_images', 'stat_image')]),
            (threshold, datasink_groupanalysis, [('thresholded_map', f"l2_analysis_{method}_nsub_{n_sub}.@thresh")]),
            (estimate_model, datasink_groupanalysis, [('mask_image', f"l2_analysis_{method}_nsub_{n_sub}.@mask")]),
            (estimate_contrast, datasink_groupanalysis, [('spm_mat_file', f"l2_analysis_{method}_nsub_{n_sub}.@spm_mat"),
                ('spmT_images', f"l2_analysis_{method}_nsub_{n_sub}.@T"),
                ('con_images', f"l2_analysis_{method}_nsub_{n_sub}.@con")])])
        
        
        if method=='equalRange' or method=='equalIndifference':
            contrasts = [('Group', 'T', ['mean'], [1]), ('Group', 'T', ['mean'], [-1])] 
            ## Specify design matrix 
            one_sample_t_test_design = Node(OneSampleTTestDesign(use_implicit_threshold=True), name = "one_sample_t_test_design")

            l2_analysis.connect([(sub_contrasts, one_sample_t_test_design, [(f"{method}_files", 'in_files'), 
                                                                           (f"{method}_covar", "covariates")]),
                                 (gunzip_mask, one_sample_t_test_design, [('out_file', 'explicit_mask_file')]),
                (one_sample_t_test_design, estimate_model, [('spm_mat_file', 'spm_mat_file')])])
            
            threshold.inputs.contrast_index = [1, 2]
            threshold.synchronize = True

        elif method == 'groupComp':
            contrasts = [('Eq range vs Eq indiff in loss', 'T', ['Group_{1}', 'Group_{2}'], [1, -1])]
            # Node for the design matrix
            two_sample_t_test_design = Node(TwoSampleTTestDesign(unequal_variance = True, use_implicit_threshold=True), 
                                            name = 'two_sample_t_test_design')

            l2_analysis.connect([(sub_contrasts, two_sample_t_test_design, [('equalRange_files', "group1_files"), 
                ('equalIndifference_files', 'group2_files'), ('global_covar', 'covariates')]),
                (gunzip_mask, two_sample_t_test_design, [('out_file', 'explicit_mask_file')]),
                (two_sample_t_test_design, estimate_model, [("spm_mat_file", "spm_mat_file")])])
            
            threshold.inputs.contrast_index = [1]
            threshold.synchronize = True

        estimate_contrast.inputs.contrasts = contrasts

        return l2_analysis
