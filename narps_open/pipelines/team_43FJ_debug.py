#!/usr/bin/python
# coding: utf-8

from os.path import join

# [INFO] The import of base objects from Nipype, to create Workflows
from nipype import Node, Workflow, MapNode

# [INFO] a list of interfaces used to manpulate data
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
# from nipype.algorithms.misc import Gunzip

# [INFO] a list of FSL-specific interfaces
from nipype.interfaces.fsl import (BET, FAST, MCFLIRT, FLIRT, FNIRT, ApplyWarp, SUSAN, 
                                   Info, ImageMaths, IsotropicSmooth, Threshold, Level1Design, FEATModel, 
                                   L2Model, Merge, FLAMEO, ContrastMgr,Cluster,  FILMGLS, Randomise, 
                                   MultipleRegressDesign, ImageStats, ExtractROI, PlotMotionParams, MeanImage,
                                   MotionOutliers)
from nipype.algorithms.modelgen import SpecifyModel
from niflow.nipype1.workflows.fmri.fsl import create_reg_workflow, create_featreg_preproc

# [INFO] In order to inherit from Pipeline
from narps_open.pipelines import Pipeline

class PipelineTeam43FJ(Pipeline):
    """ A class that defines the pipeline of team 43FJ """

    def __init__(self):
        super().__init__()
        self.team_id = '43FJ'
        self.contrast_list = ['0001', '0002', '0003', '0004']

    def get_preprocessing(self):
        """ Return a Nipype workflow describing the prepocessing part of the pipeline """

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
            SelectFiles(file_templates, base_directory = self.directories.dataset_dir),
            name='select_files'
        )

        # DataSink Node - store the wanted results in the wanted repository
        data_sink = Node(
            DataSink(base_directory = self.directories.output_dir),
            name='data_sink',
        )

        img2float = Node(
            ImageMaths(out_data_type='float', op_string='', suffix='_dtype'),
            name='img2float',
        )
    
        extract_mean = Node(
            MeanImage(), 
            name = 'extract_mean',
        )
        
        reg = create_reg_workflow()
        reg.inputs.inputspec.target_image = Info.standard_image('MNI152_T1_2mm_brain.nii.gz')
        reg.inputs.inputspec.target_image_brain = Info.standard_image('MNI152_T1_2mm_brain.nii.gz')
        reg.inputs.mean2anat.dof = 12
        reg.inputs.mean2anatbbr.dof = 12
        reg.inputs.anat2target_linear.dof = 12

        outliers = Node(MotionOutliers(metric='fd'), name='outliers')

        
        mc_smooth = create_featreg_preproc(
            name='featpreproc',
            highpass=True,
            whichvol='middle',
            whichrun=0)
        
        mc_smooth.inputs.inputspec.fwhm = 5
        mc_smooth.inputs.inputspec.highpass = 100

        # [INFO] The following part defines the nipype workflow and the connections between nodes

        preprocessing = Workflow(
            base_dir = self.directories.working_dir,
            name = 'preprocessing'
        )

        preprocessing.connect(
            [
                (
                    info_source, 
                    select_files, 
                    [('subject_id', 'subject_id'), 
                    ('run_id', 'run_id')],
                ),
                (
                    select_files, 
                    img2float, 
                    [('func', 'in_file')],
                ),
                (
                    img2float, 
                    extract_mean, 
                    [('out_file', 'in_file')],
                ),
                (
                    extract_mean, 
                    reg, 
                    [('out_file', 'inputspec.mean_image')],
                ),
                (
                    select_files, 
                    reg, 
                    [('func', 'inputspec.source_files'), ('anat', 'inputspec.anatomical_image')],
                ),
                (
                    select_files, 
                    mc_smooth, 
                    [('func', 'inputspec.func')],
                ), 
                (
                    mc_smooth, 
                    outliers, 
                    [('img2float.out_file', 'in_file')]
                ),

                (
                    reg, 
                    data_sink, 
                    [('outputspec.anat2target_transform', 'preprocess.@transfo_all'), 
                    ('outputspec.func2anat_transform', 'preprocess.@transfo_init')],
                ), 
                (
                    mc_smooth, 
                    data_sink, 
                    [('outputspec.motion_parameters', 'preprocess.@parameters_file'),
                    ('outputspec.highpassed_files', 'preprocess.@hp'),
                    ('outputspec.smoothed_files', 'preprocess.@smooth'), 
                    ('outputspec.mask', 'preprocess.@mask')],
                ),
                (
                    outliers,
                    data_sink,
                    [('out_file', 'preprocess.@outliers')]
                )
            ]
        )        

        # [INFO] Here we simply return the created workflow
        return preprocessing

    def get_preprocessing_outputs(self):
        """ Return the names of the files the preprocessing is supposed to generate. """

        templates = [join(
            self.directories.output_dir,
            'l1_analysis', f'run_id_{run_id}'+'_subject_id_{subject_id}', '_addmean0',
            'sub-{subject_id}_'+f'task-MGT_run-{run_id}_bold_dtype_mcf_mask_smooth_mask_gms_tempfilt_maths.nii.gz')\
            for run_id in self.run_list]

        templates += [join(
            self.directories.output_dir,
            'l1_analysis', f'run_id_{run_id}'+'_subject_id_{subject_id}', '_dilatemask0',
            'sub-{subject_id}_'+f'task-MGT_run-{run_id}_bold_dtype_mcf_bet_thresh_dil.nii.gz')\
            for run_id in self.run_list]

        templates += [join(
            self.directories.output_dir,
            'l1_analysis', f'run_id_{run_id}'+'_subject_id_{subject_id}', '_maskfunc30',
            'sub-{subject_id}_'+f'task-MGT_run-{run_id}_bold_dtype_mcf_mask_smooth_mask.nii.gz')\
            for run_id in self.run_list]

        templates += [join(
            self.directories.output_dir,
            'l1_analysis', f'run_id_{run_id}'+'_subject_id_{subject_id}', '_realign0',
            'sub-{subject_id}_'+f'task-MGT_run-{run_id}_bold_dtype_mcf.nii.gz.par')\
            for run_id in self.run_list]

        templates += [join(
            self.directories.output_dir,
            'l1_analysis', f'run_id_{run_id}'+'_subject_id_{subject_id}', 
            'sub-{subject_id}_'+f'T1w_fieldwarp.nii.gz')\
            for run_id in self.run_list]

        templates += [join(
            self.directories.output_dir,
            'l1_analysis', f'run_id_{run_id}'+'_subject_id_{subject_id}', 
            'sub-{subject_id}_'+f'task-MGT_run-{run_id}_bold_dtype_mean_flirt.mat')\
            for run_id in self.run_list]

        # Format with subject_ids
        return_list = []
        for template in templates:
            return_list += [template.format(subject_id = s) for s in self.subject_list]

        return return_list

    # [INFO] This function is used in the subject level analysis pipelines using FSL
    def get_subject_infos(event_file: str):
        """
        Create Bunchs for specifyModel.

        Parameters :
        - event_file : file corresponding to the run and the subject to analyze

        Returns :
        - subject_info : list of Bunch for 1st level analysis.
        """

        from os.path import join as opj
        from nipype.interfaces.base import Bunch
        
        cond_names = ['trial', 'gain', 'loss']
        
        onset = {}
        duration = {}
        amplitude = {}
        
        for c in cond_names:  # For each condition.
            onset.update({c : []}) # creates dictionary items with empty lists
            duration.update({c : []}) 
            amplitude.update({c : []})

        with open(event_file, 'rt') as f:
            next(f)  # skip the header

            for line in f:
                info = line.strip().split()
                # Creates list with onsets, duration and loss/gain for amplitude (FSL)
                for c in cond_names:
                    onset[c].append(float(info[0]))
                    duration[c].append(float(4))
                    if c == 'gain':
                        amplitude[c].append(float(info[2]))
                    elif c == 'loss':
                        amplitude[c].append(float(info[3]))
                    elif c == 'trial':
                        amplitude[c].append(float(1))
                    elif c == 'response':
                        if info[5] == 'weakly_accept':
                            amplitude[c].append(float(1))
                        elif info[5] == 'strongly_accept':
                            amplitude[c].append(float(1))
                        elif info[5] == 'weakly_reject':
                            amplitude[c].append(float(0))
                        elif info[5] == 'strongly_reject':
                            amplitude[c].append(float(0))
                        else:
                            amplitude[c].append(float(0))
                            

        subject_info = []

        subject_info.append(Bunch(
            conditions=cond_names,
            onsets=[onset[k] for k in cond_names],
            durations=[duration[k] for k in cond_names],
            amplitudes=[amplitude[k] for k in cond_names],
            regressor_names=None,
            regressors=None)
        )

        return subject_info

    # [INFO] This function creates the contrasts that will be analyzed in the first level analysis
    def get_contrasts():
        '''
        Create the list of tuples that represents contrasts. 
        Each contrast is in the form : 
        (Name,Stat,[list of condition names],[weights on those conditions])

        Parameters:
            - subject_id: str, ID of the subject 

        Returns:
            - contrasts: list of tuples, list of contrasts to analyze
        '''
        # list of condition names     
        conditions = ['gain', 'loss']
        
        # create contrasts
        gain = ('gain', 'T', conditions, [1, 0])
        
        loss = ('loss', 'T', conditions, [0, 1])
        
        # contrast list
        contrasts = [gain, loss]
        
        return contrasts

    def get_run_level_analysis(self):
        """
        Returns the first level analysis workflow.

        Parameters: 
            - exp_dir: str, directory where raw data are stored
            - result_dir: str, directory where results will be stored
            - working_dir: str, name of the sub-directory for intermediate results
            - output_dir: str, name of the sub-directory for final results
            - subject_list: list of str, list of subject for which you want to do the analysis
            - run_list: list of str, list of runs for which you want to do the analysis 
            - TR: float, time repetition used during acquisition

        Returns: 
            - l1_analysis : Nipype WorkFlow 
        """
        # Infosource Node - To iterate on subject and runs 
        info_source = Node(
            IdentityInterface(
                fields = ['subject_id', 'run_id']
            ), 
            name = 'info_source'
        )

        info_source.iterables = [
            ('subject_id', self.subject_list),
            ('run_id', self.run_list)
        ]

        templates = {
            'func': join(
                self.directories.output_dir,
                'preprocess',
                '_run_id_{run_id}_subject_id_{subject_id}','_addmean0',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_dtype_mcf_mask_smooth_mask_gms_tempfilt_maths.nii.gz',
            ),
            'event': join(
                self.directories.dataset_dir,
                'sub-{subject_id}',
                'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_events.tsv',
            ),
            'param': join(
                self.directories.output_dir,
                'preprocess',
                '_run_id_{run_id}_subject_id_{subject_id}', '_realign0',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_dtype_mcf.nii.gz.par'
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
                input_names = ['event_file'],
                output_names = ['subject_info'],
                function = self.get_subject_infos,
            ),
            name = 'subject_infos',
        )
        subject_infos.inputs.runs = self.run_list

        # Contrasts node - to get contrasts
        contrasts = Node(
            Function(
                output_names = ['contrasts'],
                function = self.get_contrasts,
            ),
            name = 'contrasts',
        )
        specify_model = Node(
            SpecifyModel(
                high_pass_filter_cutoff = 100,                         
                input_units = 'secs',
                time_repetition = 1,
            ), 
            name = 'specify_model'
        )

        l1_design = Node(
            Level1Design(
                bases = {'dgamma':{'derivs' : True}},
                interscan_interval = 1, 
                model_serial_correlations = True,
            ), 
            name = 'l1_design'
        )

        model_generation = Node(
            FEATModel(
            ), 
            name = 'model_generation'
        )

        model_estimate = Node(FILMGLS(
            ), 
            name='model_estimate'
        )

        # Create l1 analysis workflow and connect its nodes
        l1_analysis = Workflow(
            base_dir = self.directories.working_dir, 
            name = "run_level_analysis"
        )

        l1_analysis.connect([
            (
                info_source, 
                select_files, 
                [('subject_id', 'subject_id'),
                ('run_id', 'run_id')]
            ),
            (
                select_files, 
                subject_infos, 
                [('event', 'event_file')]
            ),
            (
                select_files, 
                specify_model, 
                [('param', 'realignment_parameters')]
            ),
            (
                select_files, 
                specify_model, 
                [('func', 'functional_runs')]
            ),
            (
                subject_infos, 
                specify_model, 
                [('subject_info', 'subject_info')]
            ),
            (
                contrasts, 
                l1_design, 
                [('contrasts', 'contrasts')]
            ),
            (
                specify_model, 
                l1_design, 
                [('session_info', 'session_info')]
            ),
            (
                l1_design, 
                model_generation, 
                [('ev_files', 'ev_files'), 
                ('fsf_files', 'fsf_file')]
            ),
            (
                select_files, 
                model_estimate, 
                [('func', 'in_file')]
            ),
            (
                model_generation, 
                model_estimate, 
                [('con_file', 'tcon_file'), 
                ('design_file', 'design_file')]
            ),
            (
                model_estimate, 
                data_sink, 
                [('results_dir', 'run_level_analysis.@results')]
            ),
            (
                model_generation, 
                data_sink, 
                [('design_file', 'run_level_analysis.@design_file'),
                ('design_image', 'run_level_analysis.@design_img')]
            )
        ])

        return l1_analysis

    def get_registration(self):
        """ Return a Nipype workflow describing the registration part of the pipeline """
        
        # Infosource Node - To iterate on subjects
        info_source = Node(
            IdentityInterface(
                fields = ['subject_id', 'contrast_id', 'run_id'],
            ),
            name='info_source',
        )
        info_source.iterables = [('subject_id', self.subject_list), 
                                 ('contrast_id', self.contrast_list),
                                ('run_id', self.run_list)]

        # Templates to select files node
        # [TODO] Change the name of the files depending on the filenames of results of preprocessing
        templates = {
            'cope': join(
                self.directories.output_dir,
                'run_level_analysis',
                '_run_id_{run_id}_subject_id_{subject_id}',
                'results', 
                'cope{contrast_id}.nii.gz',
            ),
            'varcope': join(
                self.directories.output_dir,
                'run_level_analysis',
                '_run_id_{run_id}_subject_id_{subject_id}',
                'results', 
                'varcope{contrast_id}.nii.gz',
            ),
            'func2anat_transform':join(
                self.directories.output_dir, 
                'preprocess', 
                '_run_id_{run_id}_subject_id_{subject_id}', 
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_dtype_mean_flirt.mat'
            ),
            'anat2target_transform':join(
                self.directories.output_dir, 
                'preprocess', 
                '_run_id_{run_id}_subject_id_{subject_id}',  
                'sub-{subject_id}_T1w_fieldwarp.nii.gz'
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
        
        warpall_cope = MapNode(
            ApplyWarp(interp='spline'),
            name='warpall_cope', 
            iterfield=['in_file']
        )
        
        warpall_cope.inputs.ref_file = Info.standard_image('MNI152_T1_2mm_brain.nii.gz')
        warpall_cope.inputs.mask_file = Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz')

        warpall_varcope = MapNode(
            ApplyWarp(interp='spline'),
            name='warpall_varcope', 
            iterfield=['in_file']
        )
        
        warpall_varcope.inputs.ref_file = Info.standard_image('MNI152_T1_2mm_brain.nii.gz')
        warpall_varcope.inputs.mask_file = Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz')

        # Create registration workflow and connect its nodes
        registration = Workflow(
            base_dir = self.directories.working_dir, 
            name = "registration"
        )

        registration.connect([
            (
                info_source, 
                select_files, 
                [('subject_id', 'subject_id'),
                ('run_id', 'run_id'),
                ('contrast_id', 'contrast_id')]
            ),
            (
                select_files, 
                warpall_cope, 
                [('func2anat_transform', 'premat'), 
                ('anat2target_transform', 'field_file'), 
                ('cope', 'in_file')]
            ), 
            (
                select_files, 
                warpall_varcope, 
                [('func2anat_transform', 'premat'), 
                ('anat2target_transform', 'field_file'), 
                ('varcope', 'in_file')]
            ), 
            (
                warpall_cope, 
                data_sink, 
                [('out_file', 'registration.@reg_cope')]
            ),
            (
                warpall_varcope, 
                data_sink, 
                [('out_file', 'registration.@reg_varcope')]
            )
        ])
        
        return registration
        
        
    def get_subject_level_analysis(self):
        """ Return a Nipype workflow describing the subject level analysis part of the pipeline """

        # [INFO] The following part stays the same for all pipelines

        # Infosource Node - To iterate on subjects
        info_source = Node(
            IdentityInterface(
                fields = ['subject_id', 'contrast_id'],
            ),
            name='info_source',
        )
        info_source.iterables = [('subject_id', self.subject_list), ('contrast_id', self.contrast_list)]

        # Templates to select files node
        # [TODO] Change the name of the files depending on the filenames of results of preprocessing
        templates = {
            'cope': join(
                self.directories.output_dir,
                'registration',
                '_contrast_id_{contrast_id}_run_id_*_subject_id_{subject_id}',
                '_warpall_cope0', 
                'cope{contrast_id}_warp.nii.gz',
            ),
            'varcope': join(
                self.directories.output_dir,
                'registration',
                '_contrast_id_{contrast_id}_run_id_*_subject_id_{subject_id}',
                '_warpall_varcope0', 
                'varcope{contrast_id}_warp.nii.gz',
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

        # Generate design matrix
        specify_model = Node(L2Model(num_copes = len(self.run_list)), name='l2model')

        # Merge copes and varcopes files for each subject
        merge_copes = Node(Merge(dimension='t'), name='merge_copes')

        merge_varcopes = Node(Merge(dimension='t'), name='merge_varcopes')

        # Second level (single-subject, mean of all four scans) analyses: Fixed effects analysis.
        flame = Node(FLAMEO(run_mode = 'fe', mask_file = Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz')), 
                     name='flameo')

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
                [('subject_id', 'subject_id'),
                 ('contrast_id', 'contrast_id')]
            ),
            (
                select_files, 
                merge_copes, 
                [('cope', 'in_files')]
            ),
            (
                select_files, 
                merge_varcopes, 
                [('varcope', 'in_files')]
            ),
            (
                merge_copes, 
                flame, 
                [('merged_file', 'cope_file')]
            ),
            (
                merge_varcopes, 
                flame, 
                [('merged_file', 'var_cope_file')]
            ),
            (
                specify_model,
                flame,
                [('design_mat', 'design_file'), 
                ('design_con', 't_con_file'),
                ('design_grp', 'cov_split_file')]
            ),
            (
                flame, 
                data_sink, 
                [('zstats', 'subject_level_analysis.@stats'),
                ('tstats', 'subject_level_analysis.@tstats'),
                ('copes', 'subject_level_analysis.@copes'),
                ('var_copes', 'subject_level_analysis.@varcopes')]
            ),
        ])

        # [INFO] Here we simply return the created workflow
        return subject_level_analysis

    # [INFO] This function returns the list of ids and files of each group of participants
    # to do analyses for both groups, and one between the two groups.
    def get_subgroups_contrasts(
        copes, varcopes, subject_list: list, participants_file: str
    ):
        """
        This function return the file list containing only the files
        belonging to subject in the wanted group.

        Parameters :
        - copes: original file list selected by select_files node
        - varcopes: original file list selected by select_files node
        - subject_ids: list of subject IDs that are analyzed
        - participants_file: file containing participants characteristics

        Returns :
        - copes_equal_indifference : a subset of copes corresponding to subjects
        in the equalIndifference group
        - copes_equal_range : a subset of copes corresponding to subjects
        in the equalRange group
        - copes_global : a list of all copes
        - varcopes_equal_indifference : a subset of varcopes corresponding to subjects
        in the equalIndifference group
        - varcopes_equal_range : a subset of varcopes corresponding to subjects
        in the equalRange group
        - equal_indifference_id : a list of subject ids in the equalIndifference group
        - equal_range_id : a list of subject ids in the equalRange group
        - varcopes_global : a list of all varcopes
        """

        equal_range_id = []
        equal_indifference_id = []

        # Reading file containing participants IDs and groups
        with open(participants_file, 'rt') as file:
            next(file)  # skip the header

            for line in file:
                info = line.strip().split()

                # Checking for each participant if its ID was selected
                # and separate people depending on their group
                if info[0][-3:] in subject_list and info[1] == 'equalIndifference':
                    equal_indifference_id.append(info[0][-3:])
                elif info[0][-3:] in subject_list and info[1] == 'equalRange':
                    equal_range_id.append(info[0][-3:])

        copes_equal_indifference = []
        copes_equal_range = []
        copes_global = []
        varcopes_equal_indifference = []
        varcopes_equal_range = []
        varcopes_global = []

        # Checking for each selected file if the corresponding participant was selected
        # and add the file to the list corresponding to its group
        for cope, varcope in zip(copes, varcopes):
            sub_id = cope.split('/')
            if sub_id[-2][-3:] in equal_indifference_id:
                copes_equal_indifference.append(cope)
            elif sub_id[-2][-3:] in equal_range_id:
                copes_equal_range.append(cope)
            if sub_id[-2][-3:] in subject_list:
                copes_global.append(cope)

            sub_id = varcope.split('/')
            if sub_id[-2][-3:] in equal_indifference_id:
                varcopes_equal_indifference.append(varcope)
            elif sub_id[-2][-3:] in equal_range_id:
                varcopes_equal_range.append(varcope)
            if sub_id[-2][-3:] in subject_list:
                varcopes_global.append(varcope)

        return copes_equal_indifference, copes_equal_range, varcopes_equal_indifference, varcopes_equal_range,equal_indifference_id, equal_range_id,copes_global, varcopes_global


    # [INFO] This function creates the dictionary of regressors used in FSL Nipype pipelines
    def get_regressors(
        equal_range_id: list,
        equal_indifference_id: list,
        method: str,
        subject_list: list,
    ) -> dict:
        """
        Create dictionary of regressors for group analysis.

        Parameters:
            - equal_range_id: ids of subjects in equal range group
            - equal_indifference_id: ids of subjects in equal indifference group
            - method: one of "equalRange", "equalIndifference" or "groupComp"
            - subject_list: ids of subject for which to do the analysis

        Returns:
            - regressors: regressors used to distinguish groups in FSL group analysis
        """
        # For one sample t-test, creates a dictionary
        # with a list of the size of the number of participants
        if method == 'equalRange':
            regressors = dict(group_mean = [1 for i in range(len(equal_range_id))])
            group = [1 for i in equal_range_id]
        elif method == 'equalIndifference':
            regressors = dict(group_mean = [1 for i in range(len(equal_indifference_id))])
            group = [1 for i in equal_indifference_id]

        # For two sample t-test, creates 2 lists:
        #  - one for equal range group,
        #  - one for equal indifference group
        # Each list contains n_sub values with 0 and 1 depending on the group of the participant
        # For equalRange_reg list --> participants with a 1 are in the equal range group
        elif method == 'groupComp':
            equalRange_reg = [
                1 for i in range(len(equal_range_id) + len(equal_indifference_id))
            ]
            equalIndifference_reg = [
                0 for i in range(len(equal_range_id) + len(equal_indifference_id))
            ]

            for index, subject_id in enumerate(subject_list):
                if subject_id in equal_indifference_id:
                    equalIndifference_reg[index] = 1
                    equalRange_reg[index] = 0

            regressors = dict(
                equalRange = equalRange_reg,
                equalIndifference = equalIndifference_reg
            )

            group = [1 if i == 1 else 2 for i in equalRange_reg]

        return regressors, group

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
        # [INFO] The following part stays the same for all preprocessing pipelines

        # Infosource node - iterate over the list of contrasts generated
        # by the subject level analysis
        info_source = Node(
            IdentityInterface(
                fields = ['contrast_id', 'subjects'],
                subjects = self.subject_list
            ),
            name = 'info_source',
        )
        info_source.iterables = [('contrast_id', self.contrast_list)]

        # Templates to select files node
        # [TODO] Change the name of the files depending on the filenames
        # of results of first level analysis
        template = {
            'cope' : join(
                self.directories.results_dir,
                'subject_level_analysis',
                '_contrast_id_{contrast_id}_subject_id_*', 'cope1.nii.gz'),
            'varcope' : join(
                self.directories.results_dir,
                'subject_level_analysis',
                '_contrast_id_{contrast_id}_subject_id_*', 'varcope1.nii.gz'),
            'participants' : join(
                self.directories.dataset_dir,
                'participants.tsv')
        }
        select_files = Node(
            SelectFiles(
                templates,
                base_directory = self.directories.results_dir,
                force_list = True
            ),
            name = 'select_files',
        )

        # Datasink node - to save important files
        data_sink = Node(
            DataSink(base_directory = self.directories.output_dir),
            name = 'data_sink',
        )

        contrasts = Node(
            Function(
                input_names=['copes', 'varcopes', 'subject_ids', 'participants_file'],
                output_names=[
                    'copes_equalIndifference',
                    'copes_equalRange',
                    'varcopes_equalIndifference',
                    'varcopes_equalRange',
                    'equalIndifference_id',
                    'equalRange_id',
                    'copes_global',
                    'varcopes_global'
                ],
                function = self.get_subgroups_contrasts,
            ),
            name = 'subgroups_contrasts',
        )

        regs = Node(
            Function(
                input_names = [
                    'equalRange_id',
                    'equalIndifference_id',
                    'method',
                    'subject_list',
                ],
                output_names = [
                'regressors',
                'group'
                ],
                function = self.get_regressors,
            ),
            name = 'regs',
        )
        regs.inputs.method = method
        regs.inputs.subject_list = self.subject_list

        # [INFO] The following part has to be modified with nodes of the pipeline

        # [TODO] For each node, replace 'node_name' by an explicit name, and use it for both:
        #   - the name of the variable in which you store the Node object
        #   - the 'name' attribute of the Node
        # [TODO] The node_function refers to a NiPype interface that you must import
        # at the beginning of the file.
        merge_copes = Node(
            Merge(dimension = 't'),
            name = 'merge_copes'
        )
        
        merge_varcopes = Node(
            Merge(dimension = 't'),
            name = 'merge_varcopes'
        )
        
        specify_model = Node(
            MultipleRegressDesign(), 
            name = 'specify_model'
        )

        flame = Node(
            FLAMEO(
                run_mode = 'flame1', 
                mask_file = Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz')
            ), 
            name='flame'
        )
        
        cluster = MapNode(
            Cluster(
                threshold = 3.1, 
                out_threshold_file = True
            ), 
            name = 'cluster', 
            iterfield = ['in_file', 'cope_file'], 
            synchronize = True
        )

        # [INFO] The following part defines the nipype workflow and the connections between nodes

        # Compute the number of participants used to do the analysis
        nb_subjects = len(self.subject_list)

        # Declare the workflow
        group_level_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = f'group_level_analysis_{method}_nsub_{nb_subjects}'
        )
        group_level_analysis.connect(
            [
                (
                    info_source,
                    select_files,
                    [('contrast_id', 'contrast_id')],
                ),
                (
                    info_source,
                    subgroups_contrasts,
                    [('subject_list', 'subject_ids')],
                ),
                (
                    select_files,
                    subgroups_contrasts,
                    [
                        ('cope', 'copes'),
                        ('varcope', 'varcopes'),
                        ('participants', 'participants_file'),
                    ],
                ),
                (
                    subgroups_contrasts, 
                    regs, 
                    [
                        ('equalRange_id', 'equalRange_id'),
                        ('equalIndifference_id', 'equalIndifference_id')
                    ]
                ),
                (
                    regs, 
                    specify_model, 
                    [('regressors', 'regressors')]
                )
            ]
        ) # Complete with other links between nodes
        
        

        # [INFO] Here we define the contrasts used for the group level analysis, depending on the
        # method used.
        if method in ('equalRange', 'equalIndifference'):
            contrasts = [('Group', 'T', ['mean'], [1]), ('Group', 'T', ['mean'], [-1])]
            
            if method == 'equalIndifference':
                group_level_analysis.connect([
                    (
                        subgroups_contrasts, 
                        merge_copes, 
                        [('copes_equalIndifference', 'in_files')]
                    ), 
                    (
                        subgroups_contrasts, 
                        merge_varcopes, 
                        [('varcopes_equalIndifference', 'in_files')]
                    )
                ])
                
            elif method == 'equalRange':
                group_level_analysis.connect([
                    (
                        subgroups_contrasts, 
                        merge_copes_3rdlevel, 
                        [('copes_equalRange', 'in_files')]
                    ),
                    (
                        subgroups_contrasts, 
                        merge_varcopes_3rdlevel, 
                        [('varcopes_equalRange', 'in_files')]
                    )
                ])

        elif method == 'groupComp':
            contrasts = [
                ('Eq range vs Eq indiff in loss', 'T', ['Group_{1}', 'Group_{2}'], [1, -1])
            ]
            
            group_level_analysis.connect([
                (
                    select_files, 
                    merge_copes, 
                    [('cope', 'in_files')]
                ),
                (
                    select_files, 
                    merge_varcopes, 
                    [('varcope', 'in_files')]
                )
            ])
            
        group_level_analysis.connect([
            (
                merge_copes, 
                flame, 
                [('merged_file', 'cope_file')]
            ),
            (
                merge_varcopes, 
                flame, 
                [('merged_file', 'var_cope_file')]
            ),
            (
                specify_model, 
                flame, 
                [
                    ('design_mat', 'design_file'),
                    ('design_con', 't_con_file'), 
                    ('design_grp', 'cov_split_file')
                ]
            ),
            (
                flame, 
                cluster, 
                [
                    ('zstats', 'in_file'), 
                    ('copes', 'cope_file')
                ]
            ),
            (
                flame,
                data_sink, 
                [
                    ('zstats', f"group_level_analysis_{method}_nsub_{nb_subjects}.@zstats"), 
                    ('tstats', f"group_level_analysis_{method}_nsub_{nb_subjects}.@tstats")
                ]
            ), 
            (
                cluster, 
                data_sink, 
                [('threshold_file', f"group_level_analysis_{method}_nsub_{nb_subjects}.@thresh")]
            )
        ])

        # [INFO] Here we simply return the created workflow
        return group_level_analysis
