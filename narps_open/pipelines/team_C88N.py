#!/usr/bin/python
# coding: utf-8

""" Write the work of NARPS' team C88N using Nipype """

#from os import
from os.path import join

from nipype import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.spm import (
    Smooth,
    OneSampleTTestDesign, EstimateModel, EstimateContrast,
    Level1Design, TwoSampleTTestDesign, Threshold
    )
from nipype.algorithms.modelgen import SpecifySPMModel
from nipype.algorithms.misc import Gunzip

from narps_open.pipelines import Pipeline

class PipelineTeamC88N(Pipeline):
    """ A class that defines the pipeline of team C88N. """

    def __init__(self):
        super().__init__()
        self.fwhm = 8.0
        self.team_id = 'C88N'
        self.model_list = ['gain', 'loss']

    def get_preprocessing(self):
        """ No preprocessing has been done by team C88N """
        return None

    def get_run_level_analysis(self):
        """ No run level analysis has been done by team C88N """
        return None

    # @staticmethod # Starting python 3.10, staticmethod should be used here
    # Otherwise it produces a TypeError: 'staticmethod' object is not callable
    def get_subject_infos(event_files: list, modulation: str):
        ''' Create Bunchs for specifySPMModel.
        Here, the team wanted to concatenate runs and used response time (RT)
        for duration except for NoResponse trials for which the duration was set to 4.
        Gain and loss amounts were used as parametric regressors.

        Parameters :
        - event_files: list of str, list of events files (one per run) for the subject
        - modulation: str, either 'gain' or 'loss'.

        Returns :
        - subject_info : list of Bunch for 1st level analysis.
        '''
        from nipype.interfaces.base import Bunch

        condition_names = ['trial']
        onset = {}
        duration = {}
        weights_gain = {}
        weights_loss = {}
        runs = ['01', '02', '03', '04']

        for run_id in range(len(runs)):  # Loop over number of runs.
            # creates dictionary items with empty lists
            onset.update({s + '_run' + str(run_id + 1) : [] for s in condition_names})
            duration.update({s + '_run' + str(run_id + 1) : [] for s in condition_names})
            weights_gain.update({'gain_run' + str(run_id + 1) : []})
            weights_loss.update({'loss_run' + str(run_id + 1) : []})

        for file_id, event_file in enumerate(event_files):
            with open(event_file, 'rt') as file:
                next(file)  # skip the header

                for line in file:
                    info = line.strip().split()

                    for condition in condition_names:
                        val = condition + '_run' + str(file_id + 1) # trial_run1
                        val_gain = 'gain_run' + str(file_id + 1) # gain_run1
                        val_loss = 'loss_run' + str(file_id + 1) # loss_run1
                        onset[val].append(float(info[0])) # onsets for trial_run1
                        duration[val].append(float(0)) # durations for trial : 0
                        weights_gain[val_gain].append(float(info[2])) # weights gain for trial_run1
                        weights_loss[val_loss].append(float(info[3])) # weights loss for trial_run1

        # Bunching is done per run, i.e. trial_run1, trial_run2, etc.
        # But names must not have '_run1' etc because we concatenate runs
        subject_info = []
        for run_id in range(len(runs)):

            conditions = [c + '_run' + str(run_id + 1) for c in condition_names]
            gain = 'gain_run' + str(run_id + 1)
            loss = 'loss_run' + str(run_id + 1)

            if modulation == 'gain':
                parametric_modulation_bunch = Bunch(
                    name = ['loss', 'gain'],
                    poly = [1, 1],
                    param = [weights_loss[loss], weights_gain[gain]]
                    )
            elif modulation == 'loss':
                parametric_modulation_bunch = Bunch(
                    name = ['gain', 'loss'],
                    poly = [1, 1],
                    param = [weights_gain[gain], weights_loss[loss]]
                    )
            else:
                raise AttributeError('')

            subject_info.insert(
                run_id,
                Bunch(
                    conditions = condition_names,
                    onsets = [onset[k] for k in conditions],
                    durations = [duration[k] for k in conditions],
                    amplitudes = None,
                    tmod = None,
                    pmod = [parametric_modulation_bunch],
                    regressor_names = None,
                    regressors = None)
                )

        return subject_info

    def get_contrasts_gain(subject_id: str):
        '''
        Create the list of tuples that represents contrasts
        Each contrast is in the form :
        (Name,Stat,[list of condition names],[weights on those conditions])

        Parameters:
            - subject_id: str, ID of the subject

        Returns:
            - contrasts: list of tuples, list of contrasts to analyze
        '''
        # List of condition names
        conditions = ['trialxloss^1', 'trialxgain^1']

        # Create contrasts
        effect_gain = ('effect_of_gain', 'T', conditions, [0, 1])

        # Return contrast list
        return [effect_gain]

    def get_contrasts_loss(subject_id: str):
        '''
        Create the list of tuples that represents contrasts.
        Each contrast is in the form :
        (Name,Stat,[list of condition names],[weights on those conditions])

        Parameters:
            - subject_id: str, ID of the subject

        Returns:
            - contrasts: list of tuples, list of contrasts to analyze
        '''
        # List of condition names
        conditions = ['trialxgain^1', 'trialxloss^1']

        # Create contrasts
        positive_effect_loss = ('positive_effect_of_loss', 'T', conditions, [0, 1])
        negative_effect_loss = ('negative_effect_of_loss', 'T', conditions, [0, -1])

        # Return contrast list
        return [positive_effect_loss, negative_effect_loss]

    # @staticmethod # Starting python 3.10, staticmethod should be used here
    # Otherwise it produces a TypeError: 'staticmethod' object is not callable
    def remove_gunzip_files(_, subject_id, working_dir):
        """
        This method is used in a Function node to fully remove
        the files generated by the gunzip node, once they aren't needed anymore.

        Parameters:
        - _: Node input only used for triggering the Node
        - subject_id: str, TODO
        - working_id: str, TODO
        """
        from shutil import rmtree
        from os.path import join

        try:
            rmtree(join(working_dir, 'l1_analysis', f'_subject_id_{subject_id}', 'gunzip_func'))
        except OSError as error:
            print(error)
        else:
            print('The directory is deleted successfully')

    # @staticmethod # Starting python 3.10, staticmethod should be used here
    # Otherwise it produces a TypeError: 'staticmethod' object is not callable
    def remove_smoothed_files(_, subject_id, working_dir):
        """
        This method is used in a Function node to fully remove
        the files generated by the smoothing node, once they aren't needed anymore.

        Parameters:
        - _: Node input only used for triggering the Node
        - subject_id: str, TODO
        - working_id: str, TODO
        """
        from shutil import rmtree
        from os.path import join

        try:
            rmtree(join(working_dir, 'l1_analysis', f'_subject_id_{subject_id}', 'smooth'))
        except OSError as error:
            print(error)
        else:
            print('The directory is deleted successfully')

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

        # Templates to select files
        template = {
            # Functional MRI
            'func' : join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-*_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'),

            # Event file
            'event' : join('sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-*_events.tsv')
        }

        # SelectFiles - to select necessary files
        selectfiles = Node(SelectFiles(template, base_directory = self.directories.dataset_dir),
            name = 'selectfiles')

        # DataSink - store the wanted results in the wanted repository
        datasink = Node(DataSink(
            base_directory = self.directories.output_dir),
            name='datasink')

        # Gunzip - gunzip files because SPM do not use .nii.gz files
        gunzip_func = MapNode(Gunzip(),
            name = 'gunzip_func',
            iterfield = ['in_file'])

        # Smooth - smoothing node
        smooth = Node(Smooth(fwhm = self.fwhm),
            name = 'smooth')

        # Funcion node get_subject_infos - get subject specific condition information
        subject_infos_gain = Node(Function(
            function = self.get_subject_infos,
            input_names = ['event_files', 'modulation'],
            output_names = ['subject_info']
            ),
            name='subject_infos_gain')
        subject_infos_gain.inputs.modulation = 'gain'

        subject_infos_loss = Node(Function(
            function = self.get_subject_infos,
            input_names = ['event_files', 'modulation'],
            output_names=['subject_info']
            ),
            name='subject_infosloss')
        subject_infos_loss.inputs.modulation = 'loss'

        # SpecifyModel - Generates SPM-specific Model
        specify_model_gain = Node(SpecifySPMModel(
            concatenate_runs = True, input_units = 'secs', output_units = 'secs',
            time_repetition = self.tr, high_pass_filter_cutoff = 128),
            name='specify_model_gain')

        specify_model_loss = Node(SpecifySPMModel(
            concatenate_runs = True, input_units = 'secs', output_units = 'secs',
            time_repetition = self.tr, high_pass_filter_cutoff = 128),
            name = 'specify_model_loss')

        # Level1Design - Generates an SPM design matrix
        l1_design_gain = Node(Level1Design(
            bases = {'hrf': {'derivs': [0, 0]}}, timing_units = 'secs',
            interscan_interval = self.tr),
            name = 'l1_design_gain')

        l1_design_loss = Node(Level1Design(
            bases = {'hrf': {'derivs': [0, 0]}}, timing_units = 'secs',
            interscan_interval = self.tr),
            name = 'l1_design_loss')

        # EstimateModel - estimate the parameters of the model
        l1_estimate_gain = Node(EstimateModel(
            estimation_method = {'Classical': 1}),
            name = "l1_estimate_gain")

        l1_estimate_loss = Node(EstimateModel(
            estimation_method = {'Classical': 1}),
            name = "l1_estimate_loss")

        # Function node get_contrasts - get the contrasts
        contrasts_gain = Node(Function(
            function = self.get_contrasts_gain,
            input_names=['subject_id'],
            output_names=['contrasts']),
            name='contrasts_gain')

        contrasts_loss = Node(Function(
            function = self.get_contrasts_loss,
            input_names=['subject_id'],
            output_names=['contrasts']),
            name='contrasts_loss')

        # EstimateContrast - estimates contrasts
        contrast_estimate_gain = Node(EstimateContrast(),
            name = "contrast_estimate_gain")

        contrast_estimate_loss = Node(EstimateContrast(),
            name = "contrast_estimate_loss")

        # Function node remove_gunzip_files - remove output of the gunzip node
        remove_gunzip_files = Node(Function(
            function = self.remove_gunzip_files,
            input_names = ['_', 'subject_id', 'working_dir'],
            output_names = []),
            name = 'remove_gunzip_files')
        remove_gunzip_files.inputs.working_dir = self.directories.working_dir

        remove_smoothed_files = Node(Function(
            function = self.remove_smoothed_files,
            input_names = ['_', 'subject_id', 'working_dir'],
            output_names = []),
            name = 'remove_smoothed_files')
        remove_smoothed_files.inputs.working_dir = self.directories.working_dir

        # Create l1 analysis workflow and connect its nodes
        l1_analysis = Workflow(base_dir = self.directories.working_dir, name = "l1_analysis")
        l1_analysis.connect([
            (infosource, selectfiles, [('subject_id', 'subject_id')]),
            (selectfiles, subject_infos_gain, [('event','event_files')]),
            (selectfiles, subject_infos_loss, [('event','event_files')]),
            (infosource, contrasts_gain, [('subject_id', 'subject_id')]),
            (infosource, contrasts_loss, [('subject_id', 'subject_id')]),
            (infosource, remove_gunzip_files, [('subject_id', 'subject_id')]),
            (infosource, remove_smoothed_files, [('subject_id', 'subject_id')]),
            (subject_infos_gain, specify_model_gain, [('subject_info', 'subject_info')]),
            (subject_infos_loss, specify_model_loss, [('subject_info', 'subject_info')]),
            (contrasts_gain, contrast_estimate_gain, [('contrasts', 'contrasts')]),
            (contrasts_loss, contrast_estimate_loss, [('contrasts', 'contrasts')]),
            (selectfiles, gunzip_func, [('func', 'in_file')]),
            (gunzip_func, smooth, [('out_file', 'in_files')]),
            (smooth, remove_gunzip_files, [('smoothed_files', '_')]),
            (smooth, specify_model_gain, [('smoothed_files', 'functional_runs')]),
            (smooth, specify_model_loss, [('smoothed_files', 'functional_runs')]),
            (specify_model_gain, l1_design_gain, [('session_info', 'session_info')]),
            (specify_model_loss, l1_design_loss, [('session_info', 'session_info')]),
            (l1_design_gain, l1_estimate_gain, [('spm_mat_file', 'spm_mat_file')]),
            (l1_design_loss, l1_estimate_loss, [('spm_mat_file', 'spm_mat_file')]),
            (l1_estimate_gain, contrast_estimate_gain, [
                ('spm_mat_file', 'spm_mat_file'),
                ('beta_images', 'beta_images'),
                ('residual_image', 'residual_image')]),
            (l1_estimate_loss, contrast_estimate_loss, [
                ('spm_mat_file', 'spm_mat_file'),
                ('beta_images', 'beta_images'),
                ('residual_image', 'residual_image')]),
            (contrast_estimate_gain, datasink, [
                ('con_images', 'l1_analysis_gain.@con_images'),
                ('spmT_images', 'l1_analysis_gain.@spmT_images'),
                ('spm_mat_file', 'l1_analysis_gain.@spm_mat_file')]),
            (contrast_estimate_loss, datasink, [
                ('con_images', 'l1_analysis_loss.@con_images'),
                ('spmT_images', 'l1_analysis_loss.@spmT_images'),
                ('spm_mat_file', 'l1_analysis_loss.@spm_mat_file')]),
            (contrast_estimate_gain, remove_smoothed_files, [('spmT_images', '_')])
            ])

        return l1_analysis

    # @staticmethod # Starting python 3.10, staticmethod should be used here
    # Otherwise it produces a TypeError: 'staticmethod' object is not callable
    def get_subset_contrasts(file_list, subject_list, participants_file):
        """
        Parameters :
        - file_list : original file list selected by selectfiles node
        - subject_list : list of subject IDs that are in the wanted group for the analysis
        - participants_file: str, file containing participants caracteristics

        This function return the file list containing only the files belonging to
        subject in the wanted group.
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

    # @staticmethod # Starting python 3.10, staticmethod should be used here
    # Otherwise it produces a TypeError: 'staticmethod' object is not callable
    def reorganize_results(team_id, nb_sub, output_dir, results_dir):
        """ Reorganize the results to analyze them.

        Parameters:
            - results_dir: str, directory where results will be stored
            - output_dir: str, name of the sub-directory for final results
            - nb_sub: float, number of subject used for the analysis
            - team_id: str, ID of the team to reorganize results
        """
        from os import mkdir
        from os.path import join, isdir
        from shutil import copyfile

        hypotheses = [
            join(output_dir,
                f'l2_analysis_equalIndifference_nsub_{nb_sub}', '_contrast_id_1_model_type_gain'),
            join(output_dir,
                f'l2_analysis_equalRange_nsub_{nb_sub}', '_contrast_id_1_model_type_gain'),
            join(output_dir,
                f'l2_analysis_equalIndifference_nsub_{nb_sub}', '_contrast_id_1_model_type_gain'),
            join(output_dir,
                f'l2_analysis_equalRange_nsub_{nb_sub}', '_contrast_id_1_model_type_gain'),
            join(output_dir,
                f'l2_analysis_equalIndifference_nsub_{nb_sub}', '_contrast_id_1_model_type_loss'),
            join(output_dir,
                f'l2_analysis_equalRange_nsub_{nb_sub}', '_contrast_id_1_model_type_loss'),
            join(output_dir,
                f'l2_analysis_equalIndifference_nsub_{nb_sub}', '_contrast_id_1_model_type_loss'),
            join(output_dir,
                f'l2_analysis_equalRange_nsub_{nb_sub}', '_contrast_id_1_model_type_loss'),
            join(output_dir,
                f'l2_analysis_groupComp_nsub_{nb_sub}', '_contrast_id_1_model_type_loss')
        ]

        # Build lists of files for unthresholded and thresholded maps
        repro_unthresh = []
        repro_thresh = []
        for file_id, filename in enumerate(hypotheses):
            if file_id in [4,5]:
                repro_unthresh.append(join(filename, 'spmT_0002.nii'))
                repro_thresh.append(join(filename, '_threshold1', 'spmT_0002_thr.nii'))
            else:
                repro_unthresh.append(join(filename, 'spmT_0001.nii'))
                repro_thresh.append(join(filename, '_threshold0', 'spmT_0001_thr.nii'))

        if not isdir(join(results_dir, "NARPS-reproduction")):
            mkdir(join(results_dir, "NARPS-reproduction"))

        for file_id, filename in enumerate(repro_unthresh):
            f_in = filename
            f_out = join(results_dir,
                'NARPS-reproduction',
                f'team_{team_id}_nsub_{nb_sub}_hypo{file_id + 1}_unthresholded.nii')
            copyfile(f_in, f_out)

        for file_id, filename in enumerate(repro_thresh):
            f_in = filename
            f_out = join(results_dir,
                'NARPS-reproduction',
                f'team_{team_id}_nsub_{nb_sub}_hypo{file_id + 1}_thresholded.nii')
            copyfile(f_in, f_out)

        print(f'Results files of team {team_id} reorganized.')

    def get_group_level_analysis(self):
        """
        Return all workflows for the group level analysis.

        Returns;
            - a list of nipype.WorkFlow
        """
        return_list = []

        self.model_list = ['gain', 'loss']
        self.contrast_list = ['0001']
        return_list.append(self.get_group_level_analysis_sub_workflow('equalRange'))
        return_list.append(self.get_group_level_analysis_sub_workflow('equalIndifference'))

        self.model_list = ['loss']
        self.contrast_list = ['0001']
        return_list.append(self.get_group_level_analysis_sub_workflow('groupComp'))

        self.model_list = ['loss']
        self.contrast_list = ['0002']
        return_list.append(self.get_group_level_analysis_sub_workflow('equalRange'))
        return_list.append(self.get_group_level_analysis_sub_workflow('equalIndifference'))

        return return_list

    def get_group_level_analysis_sub_workflow(self, method):
        """
        Return a workflow for the group level analysis.

        Parameters:
            - method: one of 'equalRange', 'equalIndifference' or 'groupComp'

        Returns:
            - l2_analysis: nipype.WorkFlow
        """
        # Compute the number of participants used to do the analysis
        nb_subjects = len(self.subject_list)

        # Infosource - iterate over the list of constrasts
        infosource_groupanalysis = Node(
            IdentityInterface(
                fields = ['model_type', 'contrast_id', 'subjects'],
                subjects = self.subject_list),
                name='infosource_groupanalysis')
        infosource_groupanalysis.iterables = [
            ('model_type', self.model_list),
            ('contrast_id', self.contrast_list)
            ]

        # SelectFiles
        templates = {
            # Contrast for all participants
            'contrasts' : join(self.directories.output_dir,
                'l1_analysis_{model_type}', '_subject_id_*', 'con_{contrast_id}.nii'),

            # Participants file
            'participants' : join(self.directories.dataset_dir, 'participants.tsv')
            }

        selectfiles_groupanalysis = Node(SelectFiles(
            templates, base_directory = self.directories.results_dir, force_list= True),
            name = 'selectfiles_groupanalysis')

        # Datasink - save important files
        datasink_groupanalysis = Node(DataSink(
            base_directory = self.directories.output_dir
            ),
            name = 'datasink_groupanalysis')

        # Function node reorganize_results - organize results once computed
        reorganize_res = Node(Function(
            function = self.reorganize_results,
            input_names = ['team_id', 'nb_subjects', 'results_dir', 'output_dir']),
            name = 'reorganize_res')
        reorganize_res.inputs.team_id = self.team_id
        reorganize_res.inputs.nb_subjects = nb_subjects
        reorganize_res.inputs.results_dir = self.directories.results_dir
        reorganize_res.inputs.output_dir = self.directories.output_dir

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

        # Estimate model
        estimate_model = Node(EstimateModel(
            estimation_method={'Classical':1}),
            name = 'estimate_model')

        # Estimate contrasts
        estimate_contrast = Node(EstimateContrast(
            group_contrast=True),
            name = 'estimate_contrast')

        # Create thresholded maps
        threshold = MapNode(Threshold(
            contrast_index = 1, use_topo_fdr = True, use_fwe_correction = False,
            extent_threshold = 0, height_threshold = 0.001, height_threshold_type = 'p-value'),
            name = 'threshold', iterfield = ['stat_image', 'contrast_index'])
        threshold.synchronize = True

        l2_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = f'l2_analysis_{method}_nsub_{nb_subjects}')
        l2_analysis.connect([
            (infosource_groupanalysis, selectfiles_groupanalysis, [
                ('contrast_id', 'contrast_id'),
                ('model_type', 'model_type')]),
            (infosource_groupanalysis, sub_contrasts, [
                ('subjects', 'subject_list')]),
            (selectfiles_groupanalysis, sub_contrasts, [
                ('contrasts', 'file_list'),
                ('participants', 'participants_file')]),
            (estimate_model, estimate_contrast, [
                ('spm_mat_file', 'spm_mat_file'),
                ('residual_image', 'residual_image'),
                ('beta_images', 'beta_images')]),
            (estimate_contrast, threshold, [
                ('spm_mat_file', 'spm_mat_file'),
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

            # Specify design matrix
            one_sample_t_test_design = Node(OneSampleTTestDesign(),
                name = 'one_sample_t_test_design')

            l2_analysis.connect([
                (sub_contrasts, one_sample_t_test_design, [(f'{method}_files', 'in_files')]),
                (one_sample_t_test_design, estimate_model, [('spm_mat_file', 'spm_mat_file')])
                ])

        elif method == 'groupComp':
            contrasts = [
                ('Eq range vs Eq indiff in loss', 'T', ['Group_{1}', 'Group_{2}'], [1, -1])]
            threshold.inputs.contrast_index = [1]

            # Specify design matrix
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
