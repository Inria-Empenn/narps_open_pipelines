#!/usr/bin/python
# coding: utf-8

"""
This template can be use to reproduce a pipeline using SPM as main software.

- Replace all occurrences of 3C6G by the actual id of the team.
- All lines starting with [INFO], are meant to help you during the reproduction, these can be removed
eventually.
- Also remove lines starting with [TODO], once you did what they suggested.
"""

# [TODO] Only import modules you use further in te code, remove others from the import section

from os.path import join

# [INFO] The import of base objects from Nipype, to create Workflows
from nipype import Node, Workflow # , JoinNode, MapNode

# [INFO] a list of interfaces used to manpulate data
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.algorithms.misc import Gunzip

# [INFO] a list of SPM-specific interfaces
from nipype.algorithms.modelgen import SpecifySPMModel
from nipype.interfaces.spm import (
   Realign, Coregister, NewSegment, Normalize12, Smooth,
   Level1Design, OneSampleTTestDesign, TwoSampleTTestDesign,
   EstimateModel, EstimateContrast, Threshold
   )
from nipype.interfaces.fsl import (
    ExtractROI
    )

# [INFO] In order to inherit from Pipeline
from narps_open.pipelines import Pipeline

class PipelineTeam3C6G(Pipeline):
    """ A class that defines the pipeline of team 3C6G """

    def __init__(self):
        super().__init__()
        # [INFO] Remove the init method completely if unused
        # [TODO] Init the attributes of the pipeline, if any other than the ones defined
        # in the pipeline class

        self.fwhm = 6.0
        self.team_id = '3C6G'
        self.contrast_list = ['0001', '0002', '0003', '0004', '0005']

    def get_vox_dims(volume : list) -> list:
        ''' 
        Function that gives the voxel dimension of an image. 
        Not used here but if we use it, modify the connection to : 
        (?, normalize_func, [('?', 'apply_to_files'),
                                        (('?', get_vox_dims),
                                         'write_voxel_sizes')])
        Args:
            volume: list | str
                List of str or str that represent a path to a Nifti image. 
        Returns: 
            list: 
                size of the voxels in the volume or in the first volume of the list.
        '''
        import nibabel as nb
        if isinstance(volume, list):
            volume = volume[0]
        nii = nb.load(volume)
        hdr = nii.header
        voxdims = hdr.get_zooms()
        return [float(voxdims[0]), float(voxdims[1]), float(voxdims[2])]

    def get_preprocessing(self):
        """ Return a Nipype workflow describing the prerpocessing part of the pipeline """

        # [INFO] The following part stays the same for all preprocessing pipelines

        # IdentityInterface node - allows to iterate over subjects and runs
        info_source = Node(
            IdentityInterface(fields=['subject_id', 'run_id']),
            name='info_source'
        )
        info_source.iterables = [
            ('subject_id', self.subject_list),
            ('run_id', self.run_list),
        ]

        # Templates to select files node
        file_templates = {
            'anat': join(
                'sub-{subject_id}', 'anat', 'sub-{subject_id}_T1w.nii.gz'
                ),
            'func': join(
                'sub-{subject_id}', 'func', 'sub-{subject_id}_task-MGT_run-{run_id}_bold.nii.gz'
                )
        }

        # SelectFiles node - to select necessary files
        select_files = Node(
            SelectFiles(
                file_templates, 
                base_directory = self.directories.dataset_dir
            ),
            name='select_files'
        )

        # DataSink Node - store the wanted results in the wanted repository
        data_sink = Node(
            DataSink(
                base_directory = self.directories.output_dir
            ),
            name='data_sink',
        )

        # [INFO] The following part has to be modified with nodes of the pipeline
        gunzip_func = Node (
            Gunzip(),
            name='gunzip_func'
        )

        gunzip_anat = Node (
            Gunzip(),
            name='gunzip_anat'
        )

        # 1 - Rigid-body realignment in SPM12 using 1st scan as referenced scan and normalized mutual information. 
        realign = Node(
            Realign(
                register_to_mean=False
            ),
            name='realign'
        )

        # Extract 1st image
        extract_first = Node(
            ExtractROI(
                t_min = 1, 
                t_size = 1,
                output_type='NIFTI'
                ),
            name = 'extract_first'
        )

        # 2 - Co-registration in SPM12 using default parameters.
        coregister = Node(
            Coregister(
                cost_function='nmi'
            ),
            name = 'coregister'
        )

        # 3 - Unified segmentation using tissue probability maps in SPM12. 
        # Unified segmentation in SPM12 to MNI space (the MNI-space tissue probability maps used in segmentation) using default parameters.
        # Bias-field correction in the context of unified segmentation in SPM12.
        tissue1 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 1), 1, (True,False), (True, False)]
        tissue2 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 2), 1, (True,False), (True, False)]
        tissue3 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 3), 2, (True,False), (True, False)]
        tissue4 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 4), 3, (True,False), (True, False)]
        tissue5 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 5), 4, (True,False), (True, False)]
        tissue6 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 6), 2, (True,False), (True, False)]
        tissue_list = [tissue1, tissue2, tissue3, tissue4, tissue5, tissue6]

        segment = Node(
            NewSegment(
                write_deformation_fields = [True, True], 
                tissues = tissue_list
            ),
            name = 'segment'
        )

        # 4 - Spatial normalization of functional images
        normalize = Node(
            Normalize12(
                jobtype = 'write'
            ),
            name = 'normalize'
        )

        # 5 - 6 mm fixed FWHM smoothing in MNI volume
        smooth = Node(
            Smooth(
                fwhm=self.fwhm),
            name = 'smooth'
        )

        # [INFO] The following part defines the nipype workflow and the connections between nodes

        preprocessing = Workflow(
            base_dir = self.directories.working_dir,
            name = 'preprocessing'
        )

        # [TODO] Add the connections the workflow needs
        # [INFO] Input and output names can be found on NiPype documentation
        preprocessing.connect(
            [
                (
                    info_source,
                    select_files,
                    [('subject_id', 'subject_id'), ('run_id', 'run_id')],
                ),
                (
                    select_files,
                    gunzip_anat, 
                    [('anat', 'in_file')]
                ),
                (
                    select_files,
                    gunzip_func, 
                    [('func', 'in_file')]
                ),
                (
                    gunzip_func,
                    realign,
                    [('out_file', 'in_files')],
                ),
                (
                    realign,
                    extract_first,
                    [('realigned_files', 'in_file')],
                ),
                (
                    extract_first,
                    coregister,
                    [('roi_file', 'source')],
                ),
                (
                    realign,
                    coregister,
                    [('realigned_files', 'apply_to_files')],
                ),
                (
                    gunzip_anat,
                    coregister,
                    [('out_file', 'target')],
                ),
                (
                    gunzip_anat,
                    segment,
                    [('out_file', 'channel_files')],
                ),
                (
                    segment,
                    normalize,
                    [('forward_deformation_field', 'deformation_file')],
                ),
                (
                    coregister,
                    normalize,
                    [('coregistered_files', 'apply_to_files')],
                ),
                (
                    normalize,
                    smooth,
                    [('normalized_files', 'in_files')],
                ),
                (
                    smooth,
                    data_sink,
                    [('smoothed_files', 'preprocessing.@smoothed')],
                ),
                (
                    realign,
                    data_sink,
                    [('realignment_parameters', 'preprocessing.@motion_parameters')],
                ),
                (
                    segment,
                    data_sink,
                    [('native_class_images', 'preprocessing.@segmented'),
                    ('normalized_class_images', 'preprocessing.@segmented_normalized')],
                ),

            ]
        )

        # [INFO] Here we simply return the created workflow
        return preprocessing

    def get_preprocessing_outputs(self):
        """ Return the names of the files the preprocessing analysis is supposed to generate. """

        # Smoothed maps
        templates = [join(
            self.directories.output_dir,
            'preprocessing', '_run_id_{run_id}_subject_id_{subject_id}', 
            'swrrsub-{subject_id}_task-MGT_run-{run_id}_bold.nii')]

        # Motion parameters file
        templates += [join(
            self.directories.output_dir,
            'preprocessing', '_run_id_{run_id}_subject_id_{subject_id}', 
            'rp_sub-{subject_id}_task-MGT_run-{run_id}_bold.txt')]

        # Segmentation maps
        templates += [join(
            self.directories.output_dir,
            'preprocessing', '_run_id_{run_id}_subject_id_{subject_id}', 
            f'c{i}'+'sub-{subject_id}_T1w.nii')\
            for i in range(1,7)]

        templates += [join(
            self.directories.output_dir,
            'preprocessing', '_run_id_{run_id}_subject_id_{subject_id}', 
            f'wc{i}'+'sub-{subject_id}_T1w.nii')\
            for i in range(1,7)]

        # Format with subject_ids
        return_list = []
        for template in templates:
            return_list += [template.format(subject_id = s, run_id = r) for r in self.run_list for s in self.subject_list]

        return return_list


    # [INFO] There was no run level analysis for the pipelines using SPM
    def get_run_level_analysis(self):
        """ Return a Nipype workflow describing the run level analysis part of the pipeline """
        return None

    # [INFO] This function is used in the subject level analysis pipelines using SPM
    # [TODO] Adapt this example to your specific pipeline
    def get_subject_infos(event_files: list, runs: list):
        """
         -MGT task (taken from .tsv files, duration = 4) with canonical HRF (no derivatives)
        -Parametric modulator gain (from "gain" column in event .tsv file)
        - Parametric modulator loss (from "loss" column in event .tsv file)
        -highpass DCT filtering in SPM (using default period of 1/128 s)
        -6 movement regressors from realignment

        Create Bunchs for specifySPMModel.

        Parameters :
        - event_files: list of events files (one per run) for the subject
        - runs: list of runs to use

        Returns :
        - subject_info : list of Bunch for 1st level analysis.
        """
        from nipype.interfaces.base import Bunch

        condition_names = ['trial']
        onset = {}
        duration = {}
        weights_gain = {}
        weights_loss = {}

        # Loop over number of runs
        for run_id in range(len(runs)):

            # Create dictionary items with empty lists
            onset.update({s + '_run' + str(run_id + 1): [] for s in condition_names})
            duration.update({s + '_run' + str(run_id + 1): [] for s in condition_names})
            weights_gain.update({'gain_run' + str(run_id + 1): []})
            weights_loss.update({'loss_run' + str(run_id + 1): []})

            with open(event_files[run_id], 'rt') as event_file:
                next(event_file)  # skip the header

                for line in event_file:
                    info = line.strip().split()

                    for condition in condition_names:
                        val = condition + '_run' + str(run_id + 1)  # trial_run1 or accepting_run1
                        val_gain = 'gain_run' + str(run_id + 1)  # gain_run1
                        val_loss = 'loss_run' + str(run_id + 1)  # loss_run1
                        if condition == 'trial':
                            onset[val].append(float(info[0]))  # onsets for trial_run1
                            duration[val].append(float(4))
                            weights_gain[val_gain].append(float(info[2]))
                            weights_loss[val_loss].append(float(info[3]))

        # Bunching is done per run, i.e. trial_run1, trial_run2, etc.
        # But names must not have '_run1' etc because we concatenate runs
        subject_info = []
        for run_id in range(len(runs)):

            conditions = [s + '_run' + str(run_id + 1) for s in condition_names]
            gain = 'gain_run' + str(run_id + 1)
            loss = 'loss_run' + str(run_id + 1)

            subject_info.insert(
                run_id,
                Bunch(
                    conditions = condition_names,
                    onsets = [onset[c] for c in conditions],
                    durations = [duration[c] for c in conditions],
                    amplitudes = None,
                    tmod = None,
                    pmod = [
                        Bunch(
                            name = ['gain', 'loss'],
                            poly = [1, 1],
                            param = [weights_gain[gain], weights_loss[loss]],
                        ),
                        None,
                    ],
                    regressor_names = None,
                    regressors = None,
                ),
            )

        return subject_info

    # [INFO] This function creates the contrasts that will be analyzed in the first level analysis
    # [TODO] Adapt this example to your specific pipeline
    def get_contrasts():
        """
        Create the list of tuples that represents contrasts.
        Each contrast is in the form :
        (Name,Stat,[list of condition names],[weights on those conditions])

        Returns:
            - contrasts: list of tuples, list of contrasts to analyze
        """
        # List of condition names
        conditions = ['trial', 'trialxgain^1', 'trialxloss^1']

        # Create contrasts
        trial = ('trial', 'T', conditions, [1, 0, 0])
        effect_gain = ('effect_of_gain', 'T', conditions, [0, 1, 0])
        neg_effect_gain = ('neg_effect_of_gain', 'T', conditions, [0, -1, 0])
        effect_loss = ('effect_of_loss', 'T', conditions, [0, 0, 1])
        neg_effect_loss = ('neg_effect_of_loss', 'T', conditions, [0, 0, -1])

        contrasts = [trial, effect_gain, effect_loss, neg_effect_gain, neg_effect_loss]

        return contrasts

    def get_subject_level_analysis(self):
        """ Return a Nipype workflow describing the subject level analysis part of the pipeline """

        # [INFO] The following part stays the same for all pipelines

        # Infosource Node - To iterate on subjects
        info_source = Node(
            IdentityInterface(
                fields = ['subject_id', 'dataset_dir', 'results_dir', 'working_dir', 'run_list'],
                dataset_dir = self.directories.dataset_dir,
                results_dir = self.directories.results_dir,
                working_dir = self.directories.working_dir,
                run_list = self.run_list
            ),
            name='info_source',
        )
        info_source.iterables = [('subject_id', self.subject_list)]

        # Templates to select files node
        # [TODO] Change the name of the files depending on the filenames of results of preprocessing
        templates = {
            'func': join(
                self.directories.results_dir,
                'preprocessing',
                '_run_id_*_subject_id_{subject_id}',
                'swrrsub-{subject_id}_task-MGT_run-*_bold.nii',
            ),
            'event': join(
                self.directories.dataset_dir,
                'sub-{subject_id}',
                'func',
                'sub-{subject_id}_task-MGT_run-*_events.tsv',
            ),
            'parameters': join(
                self.directories.results_dir,
                'preprocessing',
                '_run_id_*_subject_id_{subject_id}',
                'rp_sub-{subject_id}_task-MGT_run-*_bold.txt',
            )
        }

        # SelectFiles node - to select necessary files
        select_files = Node(
            SelectFiles(templates, base_directory = self.directories.dataset_dir),
            name = 'select_files'
        )

        # DataSink Node - store the wanted results in the wanted repository
        data_sink = Node(
            DataSink(base_directory = self.directories.output_dir),
            name = 'data_sink'
        )

        # [INFO] This is the node executing the get_subject_infos_spm function
        # Subject Infos node - get subject specific condition information
        subject_infos = Node(
            Function(
                input_names = ['event_files', 'runs'],
                output_names = ['subject_info'],
                function = self.get_subject_infos,
            ),
            name = 'subject_infos',
        )
        subject_infos.inputs.runs = self.run_list

        # [INFO] This is the node executing the get_contrasts function
        # Contrasts node - to get contrasts
        contrasts = Node(
            Function(
                output_names = ['contrasts'],
                function = self.get_contrasts,
            ),
            name = 'contrasts',
        )

        # [INFO] The following part has to be modified with nodes of the pipeline

        # [TODO] For each node, replace 'node_name' by an explicit name, and use it for both:
        #   - the name of the variable in which you store the Node object
        #   - the 'name' attribute of the Node
        # [TODO] The node_function refers to a NiPype interface that you must import
        # at the beginning of the file.
        # SpecifyModel - generates SPM-specific Model
        specify_model = Node(
            SpecifySPMModel(
            concatenate_runs = True, 
            input_units = 'secs', 
            output_units = 'secs',
            time_repetition = self.tr, 
            high_pass_filter_cutoff = 128),
            name = 'specify_model'
        )

        # Level1Design - generates an SPM design matrix
        l1_design = Node(
            Level1Design(
            bases = {'hrf': {'derivs': [0, 0]}}, 
            timing_units = 'secs',
            interscan_interval = self.tr,
            model_serial_correlations='AR(1)'),
            name = 'l1_design'
        )

        # EstimateModel - estimate the parameters of the model
        l1_estimate = Node(
            EstimateModel(
            estimation_method = {'Classical': 1}),
            name = 'l1_estimate'
        )

        # EstimateContrast - estimates contrasts
        contrast_estimate = Node(
            EstimateContrast(),
            name = 'contrast_estimate'
        )

        # [INFO] The following part defines the nipype workflow and the connections between nodes

        subject_level_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = 'subject_level_analysis'
        )
        # [TODO] Add the connections the workflow needs
        # [INFO] Input and output names can be found on NiPype documentation
        subject_level_analysis.connect([
            (
                info_source,
                select_files,
                [('subject_id', 'subject_id')]
            ),
            (
                select_files,
                subject_infos,
                [('event', 'event_files')]
            ),
            (
                subject_infos, 
                specify_model, 
                [('subject_info', 'subject_info')]
            ),
            (
                contrasts, 
                contrast_estimate, 
                [('contrasts', 'contrasts')]
            ),
            (
                select_files,
                specify_model, 
                [('func', 'functional_runs'), ('parameters', 'realignment_parameters')]
            ),
            (
                specify_model, 
                l1_design, 
                [('session_info', 'session_info')]
            ),
            (
                l1_design, 
                l1_estimate, 
                [('spm_mat_file', 'spm_mat_file')]
            ),
            (
                l1_estimate, 
                contrast_estimate, 
                [('spm_mat_file', 'spm_mat_file'),
                ('beta_images', 'beta_images'),
                ('residual_image', 'residual_image')]
            ),
            (
                contrast_estimate, 
                data_sink, 
                [('con_images', 'l1_analysis.@con_images'),
                ('spmT_images', 'l1_analysis.@spmT_images'),
                ('spm_mat_file', 'l1_analysis.@spm_mat_file')]
            ),
        ])

        # [INFO] Here we simply return the created workflow
        return subject_level_analysis

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

    # [INFO] This function returns the list of ids and files of each group of participants
    # to do analyses for both groups, and one between the two groups.
    def get_subset_contrasts(
        file_list, subject_list: list, participants_file: str
    ):
        """
        This function return the file list containing only the files belonging
        to the subjects in the wanted group.

        Parameters :
        - file_list : original file list selected by selectfiles node
        - subject_list : list of subject IDs that are in the wanted group for the analysis
        - participants_file: str, file containing participants characteristics

        Returns :
        - equal_indifference_id : a list of subject ids in the equalIndifference group
        - equal_range_id : a list of subject ids in the equalRange group
        - equal_indifference_files : a subset of file_list corresponding to subjects
        in the equalIndifference group
        - equal_range_files : a subset of file_list corresponding to subjects
        in the equalRange group
        """
        equal_indifference_id = []
        equal_range_id = []
        equal_indifference_files = []
        equal_range_files = []

        # Reading file containing participants IDs and groups
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
        Return a workflow for the group level analysis.

        Parameters:
            - method: one of 'equalRange', 'equalIndifference' or 'groupComp'

        Returns:
            - group_level_analysis: nipype.WorkFlow
        """
        # Compute the number of participants used to do the analysis
        nb_subjects = len(self.subject_list)

        # Infosource - iterate over the list of contrasts
        infosource_groupanalysis = Node(
            IdentityInterface(
                fields = ['contrast_id', 'subjects']),
                name = 'infosource_groupanalysis')
        infosource_groupanalysis.iterables = [('contrast_id', self.contrast_list)]

        # SelectFiles
        templates = {
            # Contrast for all participants
            'contrast' : join(self.directories.output_dir,
                'l1_analysis', '_subject_id_*', 'con_{contrast_id}.nii'),
            # Participants file
            'participants' : join(self.directories.dataset_dir, 'participants.tsv')
            }

        selectfiles_groupanalysis = Node(SelectFiles(
            templates, base_directory = self.directories.results_dir, force_list = True),
            name = 'selectfiles_groupanalysis')

        # Datasink - save important files
        datasink_groupanalysis = Node(DataSink(
            base_directory = str(self.directories.output_dir)
            ),
            name = 'datasink_groupanalysis')

        # Function node get_subset_contrasts - select subset of contrasts
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
            height_threshold = 0.001, height_threshold_type = 'p-value',
            extent_fdr_p_threshold = 0.05,
            force_activation = True),
            name = 'threshold', 
            iterfield = ['stat_image', 'contrast_index'])

        l2_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = f'l2_analysis_{method}_nsub_{nb_subjects}')
        l2_analysis.connect([
            (infosource_groupanalysis, selectfiles_groupanalysis, [
                ('contrast_id', 'contrast_id')]),
            (selectfiles_groupanalysis, sub_contrasts, [
                ('contrast', 'file_list'),
                ('participants', 'participants_file')]),
            (estimate_model, estimate_contrast, [('spm_mat_file', 'spm_mat_file'),
                ('residual_image', 'residual_image'),
                ('beta_images', 'beta_images')]),
            (estimate_contrast, threshold, [('spm_mat_file', 'spm_mat_file'),
                ('spmT_images', 'stat_image')]),
            (estimate_model, datasink_groupanalysis, [
                ('mask_image', f'l2_analysis_{method}_nsub_{nb_subjects}.@mask')]),
            (estimate_contrast, datasink_groupanalysis, [
                ('spm_mat_file', f'l2_analysis_{method}_nsub_{nb_subjects}.@spm_mat'),
                ('spmT_images', f'l2_analysis_{method}_nsub_{nb_subjects}.@T'),
                ('con_images', f'l2_analysis_{method}_nsub_{nb_subjects}.@con')]),
            (threshold, datasink_groupanalysis, [
                ('thresholded_map', f'l2_analysis_{method}_nsub_{nb_subjects}.@thresh')])])

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
                ('Eq range vs Eq indiff in loss', 'T', ['Group_{1}', 'Group_{2}'], [-1, 1])]

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
                '_contrast_id_0002', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'l2_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0001.nii'),
            # Hypothesis 2
            join(f'l2_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'l2_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0001.nii'),
            # Hypothesis 3
            join(f'l2_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'l2_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0001.nii'),
            # Hypothesis 4
            join(f'l2_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'l2_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0001.nii'),
            # Hypothesis 5
            join(f'l2_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0005', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'l2_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0005', 'spmT_0001.nii'),
            # Hypothesis 6
            join(f'l2_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0005', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'l2_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0005', 'spmT_0001.nii'),
            # Hypothesis 7
            join(f'l2_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0003', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'l2_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0003', 'spmT_0001.nii'),
            # Hypothesis 8
            join(f'l2_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0003', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'l2_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0003', 'spmT_0001.nii'),
            # Hypothesis 9
            join(f'l2_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_0003', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'l2_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_0003', 'spmT_0001.nii')
        ]
        return [join(self.directories.output_dir, f) for f in files]
