#!/usr/bin/python
# coding: utf-8

""" Write the work of NARPS' team Q6O0 using Nipype """

from os.path import join
from itertools import product

from nipype import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.spm import (
    Smooth,
    Level1Design, OneSampleTTestDesign, TwoSampleTTestDesign,
    EstimateModel, EstimateContrast, Threshold
    )
from nipype.algorithms.modelgen import SpecifySPMModel
from nipype.algorithms.misc import Gunzip

from narps_open.pipelines import Pipeline

class PipelineTeamQ6O0(Pipeline):
    """ A class that defines the pipeline of team Q6O0. """

    def __init__(self):
        super().__init__()
        self.fwhm = 8.0
        self.team_id = 'Q6O0'
        self.contrast_list = ['0001']
        self.model_list = ['gain', 'loss']

    def get_preprocessing(self):
        """ No preprocessing has been done by team Q6O0 """
        return None

    def get_run_level_analysis(self):
        """ No run level analysis has been done by team Q6O0 """
        return None

    # @staticmethod # Starting python 3.10, staticmethod should be used here
    # Otherwise it produces a TypeError: 'staticmethod' object is not callable
    def get_subject_infos(event_files, runs, model):
        """ Create Bunchs for specifySPMModel.
        Here, the team wanted to concatenate runs and used RT (response time)
        for duration except for NoResponse trials for which the duration was set to 4.
        Gain and loss amounts were used as parametric regressors.

        Parameters :
        - event_files : list of files containing events information for each run
        - runs: list of str, list of runs to use
        - model: str, either 'gain' or 'loss'.

        Returns :
        - subject_info : list of Bunch for 1st level analysis.
        """
        from nipype.interfaces.base import Bunch

        condition_names = ['trial']
        onset = {}
        duration = {}
        weights_gain = {}
        weights_loss = {}

        for run_id in range(len(runs)):  # Loop over number of runs.
            # creates dictionary items with empty lists
            onset.update({s + '_run' + str(run_id + 1) : [] for s in condition_names})
            duration.update({s + '_run' + str(run_id + 1) : [] for s in condition_names})
            weights_gain.update({'gain_run' + str(run_id + 1) : []})
            weights_loss.update({'loss_run' + str(run_id + 1) : []})

        for run_id, event_file in enumerate(event_files):
            with open(event_file, 'rt') as file:
                next(file)  # skip the header

                for line in file:
                    info = line.strip().split()

                    for condition in condition_names:
                        val = condition + '_run' + str(run_id + 1) # trial_run1
                        val_gain = 'gain_run' + str(run_id + 1) # gain_run1
                        val_loss = 'loss_run' + str(run_id + 1) # loss_run1
                        onset[val].append(float(info[0])) # onsets for trial_run1
                        duration[val].append(float(4)) # durations for trial : 4
                        weights_gain[val_gain].append(float(info[2])) # weights gain for trial_run1
                        weights_loss[val_loss].append(float(info[3])) # weights loss for trial_run1

        # Bunching is done per run, i.e. trial_run1, trial_run2, etc.
        # But names must not have '_run1' etc because we concatenate runs
        subject_info = []
        for run_id in range(len(runs)):

            conditions = [c + '_run' + str(run_id + 1) for c in condition_names]
            gain = 'gain_run' + str(run_id + 1)
            loss = 'loss_run' + str(run_id + 1)

            if model == 'gain':
                parametric_modulation_bunch = Bunch(
                    name = ['loss', 'gain'],
                    poly = [1, 1],
                    param = [weights_loss[loss], weights_gain[gain]]
                    )
            elif model == 'loss':
                parametric_modulation_bunch = Bunch(
                    name = ['gain', 'loss'],
                    poly = [1, 1],
                    param = [weights_gain[gain], weights_loss[loss]]
                    )
            else:
                raise AttributeError('Model must be gain or loss.')

            subject_info.insert(
                run_id,
                Bunch(
                    conditions = condition_names,
                    onsets=[onset[k] for k in conditions],
                    durations = [duration[k] for k in conditions],
                    amplitudes = None,
                    tmod = None,
                    pmod = [parametric_modulation_bunch],
                    regressor_names = None,
                    regressors = None)
                )

        return subject_info

    def get_contrasts_gain(subject_id):
        """
        Create the list of tuples that represents contrasts.
        Each contrast is in the form :
        (Name,Stat,[list of condition names],[weights on those conditions])

        Parameters:
        	- subject_id: str, ID of the subject

        Returns:
        	- contrasts: list of tuples, list of contrasts to analyze
        """
        # List of condition names
        conditions = ['trialxgain^1']

        # Create contrasts
        positive_effect_gain = ('positive_effect_gain', 'T', conditions, [1])

        # Return contrast list
        return [positive_effect_gain]

    def get_contrasts_loss(subject_id):
        """
        Create the list of tuples that represents contrasts.
        Each contrast is in the form :
        (Name,Stat,[list of condition names],[weights on those conditions])

        Parameters:
        	- subject_id: str, ID of the subject

        Returns:
        	- contrasts: list of tuples, list of contrasts to analyze
        """
        # List of condition names
        conditions = ['trialxloss^1']

        # Create contrasts
        positive_effect_loss = ('positive_effect_loss', 'T', conditions, [1])

        # Return contrast list
        return [positive_effect_loss]

    def get_parameters_file(filepaths, subject_id, working_dir):
        """
        Create new tsv files with only desired parameters per subject per run.
        The six motion parameters, the 5 aCompCor parameters, the global white matter and
        cerebral spinal fluid signals were included as nuisance regressors/

        Parameters :
        - filepaths : paths to subject parameters file (i.e. one per run)
        - subject_id : subject for whom the 1st level analysis is made
    	- working_dir: str, name of the sub-directory for intermediate results

        Return :
        - parameters_file : paths to new files containing only desired parameters.
        """
        from os import makedirs
        from os.path import join, isdir

        import pandas as pd
        import numpy as np

        # Handle the case where filepaths is a single path (str)
        if not isinstance(filepaths, list):
            filepaths = [filepaths]

        # Create the parameters files
        parameters_file = []
        for file_id, file in enumerate(filepaths):
            data_frame = pd.read_csv(file, sep = '\t', header=0)

            # Extract parameters we want to use for the model
            temp_list = np.array([
                data_frame['X'], data_frame['Y'], data_frame['Z'],
                data_frame['RotX'], data_frame['RotY'], data_frame['RotZ'],
                data_frame['aCompCor00'], data_frame['aCompCor01'], data_frame['aCompCor02'],
                data_frame['aCompCor03'], data_frame['aCompCor04'], data_frame['aCompCor05'],
                data_frame['WhiteMatter'], data_frame['CSF']])
            retained_parameters = pd.DataFrame(np.transpose(temp_list))

            # Write parameters to a parameters file
            # TODO : warning !!! filepaths must be ordered (1,2,3,4) for the following code to work
            new_path =join(working_dir, 'parameters_file',
                f'parameters_file_sub-{subject_id}_run-{str(file_id + 1).zfill(2)}.tsv')

            makedirs(join(working_dir, 'parameters_file'), exist_ok = True)

            with open(new_path, 'w') as writer:
                writer.write(retained_parameters.to_csv(
                    sep = '\t', index = False, header = False, na_rep = '0.0'))

            parameters_file.append(new_path)

        return parameters_file

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
        infosource = Node(IdentityInterface( fields = ['subject_id']),
            name = 'infosource')
        infosource.iterables = [('subject_id', self.subject_list)]

        # Templates to select files node
        template = {
            'param' : join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-*_bold_confounds.tsv'),
            'event' : join('sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-*_events.tsv'),
            'func' : join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-*_bold_space-MNI152NLin2009cAsym_preproc.nii.gz')
        }

        # SelectFiles - to select necessary files
        selectfiles = Node(SelectFiles(template, base_directory = self.directories.dataset_dir),
            name = 'selectfiles')

        # DataSink - store the wanted results in the wanted repository
        datasink = Node(DataSink(base_directory = self.directories.output_dir),
            name='datasink')

        # Gunzip - gunzip files because SPM do not use .nii.gz files
        gunzip_func = MapNode(Gunzip(), name = 'gunzip_func', iterfield = ['in_file'])

        # Smooth - smoothing node
        smooth = Node(Smooth(fwhm = self.fwhm),
            name = 'smooth')

        # Function node get_subject_infos - get subject specific condition information
        subject_infos_gain = Node(Function(
            function = self.get_subject_infos,
            input_names = ['event_files', 'runs', 'model'],
            output_names=['subject_info']),
            name='subject_infos_gain')
        subject_infos_gain.inputs.runs = self.run_list
        subject_infos_gain.inputs.model = 'gain'

        subject_infos_loss = Node(Function(
            function = self.get_subject_infos,
            input_names = ['event_files', 'runs', 'model'],
            output_names = ['subject_info']),
            name='subject_infos_loss')
        subject_infos_loss.inputs.runs = self.run_list
        subject_infos_loss.inputs.model = 'loss'

        # Function node get_parameters_file - get parameters files
        parameters = Node(Function(
            function = self.get_parameters_file,
            input_names = ['filepaths', 'subject_id', 'working_dir'],
            output_names = ['parameters_file']),
            name = 'parameters')
        parameters.inputs.working_dir = self.directories.working_dir

        # SpecifyModel - Generates SPM-specific Model
        specify_model_gain = Node(SpecifySPMModel(
            concatenate_runs = True, input_units = 'secs', output_units = 'secs',
            time_repetition = self.tr, high_pass_filter_cutoff = 128),
            name='specify_model_gain')

        specify_model_loss = Node(SpecifySPMModel(
            concatenate_runs = True, input_units = 'secs', output_units = 'secs',
            time_repetition = self.tr, high_pass_filter_cutoff = 128),
            name='specify_model_loss')

        # Level1Design - Generates an SPM design matrix
        l1_design_gain = Node(Level1Design(
            bases = {'hrf': {'derivs': [0, 0]}}, timing_units = 'secs',
            interscan_interval = self.tr, model_serial_correlations = 'AR(1)'),
            name='l1_design_gain')

        l1_design_loss = Node(Level1Design(
            bases = {'hrf': {'derivs': [0, 0]}}, timing_units = 'secs',
            interscan_interval = self.tr, model_serial_correlations = 'AR(1)'),
            name='l1_design_loss')

        # EstimateModel - estimate the parameters of the model
        l1_estimate_gain = Node(EstimateModel(
            estimation_method = {'Classical': 1}),
            name = 'l1_estimate_gain')

        l1_estimate_loss = Node(EstimateModel(
            estimation_method = {'Classical': 1}),
            name = 'l1_estimate_loss')

        # Function nodes get_contrasts_* - get the contrasts
        contrasts_gain = Node(Function(
            function = self.get_contrasts_gain,
            input_names = ['subject_id'],
            output_names = ['contrasts']),
            name = 'contrasts_gain')

        contrasts_loss = Node(Function(
            function = self.get_contrasts_loss,
            input_names = ['subject_id'],
            output_names = ['contrasts']),
            name = 'contrasts_loss')

        # EstimateContrast - estimates contrasts
        contrast_estimate_gain = Node(EstimateContrast(),
            name = 'contrast_estimate_gain')

        contrast_estimate_loss = Node(EstimateContrast(),
            name = 'contrast_estimate_loss')

        # Function node remove_gunzip_files - remove output of the gunzip node
        remove_gunzip_files = Node(Function(
            function = self.remove_gunzip_files,
            input_names = ['_', 'subject_id', 'working_dir'],
            output_names = []),
            name = 'remove_gunzip_files')
        remove_gunzip_files.inputs.working_dir = self.directories.working_dir

        # Function node remove_smoothed_files - remove output of the smooth node
        remove_smoothed_files = Node(Function(
            function = self.remove_smoothed_files,
            input_names = ['_', 'subject_id', 'working_dir'],
            output_names = []),
            name = 'remove_smoothed_files')
        remove_smoothed_files.inputs.working_dir = self.directories.working_dir

        # Create l1 analysis workflow and connect its nodes
        l1_analysis = Workflow(base_dir = self.directories.working_dir, name = 'l1_analysis')
        l1_analysis.connect([
            (infosource, selectfiles, [('subject_id', 'subject_id')]),
            (selectfiles, subject_infos_gain, [('event','event_files')]),
            (selectfiles, subject_infos_loss, [('event','event_files')]),
            (selectfiles, parameters, [('param', 'filepaths')]),
            (infosource, parameters, [('subject_id', 'subject_id')]),
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
            (parameters, specify_model_gain, [('parameters_file', 'realignment_parameters')]),
            (parameters, specify_model_loss, [('parameters_file', 'realignment_parameters')]),
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

    def get_subject_level_outputs(self):
        """ Return the names of the files the subject level analysis is supposed to generate. """

        # Generate a list of parameter sets for further templates formatting
        parameters = {
            'contrast_id': self.contrast_list,
            'model_type': self.model_list,
            'subject_id': self.subject_list
        }
        # Combining all possibilities
        # Here we use a list because the itertools.product is an iterator objects that
        # is meant for a single-use iteration only.
        parameter_sets = list(product(*parameters.values()))

        # Contrat maps
        contrast_map_template = join(
            self.directories.output_dir,
            'l1_analysis_{model_type}', '_subject_id_{subject_id}', 'con_{contrast_id}.nii'
            )

        # SPM.mat file
        mat_file_template = join(
            self.directories.output_dir,
            'l1_analysis_{model_type}', '_subject_id_{subject_id}', 'SPM.mat'
            )

        # spmT maps
        spmt_file_template = join(
            self.directories.output_dir,
            'l1_analysis_{model_type}', '_subject_id_{subject_id}', 'spmT_{contrast_id}.nii'
            )

        # Formatting templates and returning it as a list of files
        output_files = [contrast_map_template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]
        output_files += [mat_file_template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]
        output_files += [spmt_file_template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

        return output_files

    def get_subset_contrasts(file_list, subject_list, participants_file):
        """
        Parameters :
        - file_list : original file list selected by selectfiles node
        - subject_list : list of subject IDs that are in the wanted group for the analysis
        - participants_file: str, file containing participants characteristics

        Returns:
        - The file list containing only the files belonging to subject in the wanted group.
        """
        equal_indifference_id = []
        equal_range_id = []
        equal_indifference_files = []
        equal_range_files = []

        with open(participants_file, 'rt') as file:
            next(file)  # skip the header
            for line in file:
                info = line.strip().split()
                if info[0][-3:] in subject_list and info[1] == "equalIndifference":
                    equal_indifference_id.append(info[0][-3:])
                elif info[0][-3:] in subject_list and info[1] == "equalRange":
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
            - l2_analysis: nipype.WorkFlow
        """
        # Compute the number of participants used to do the analysis
        nb_subjects = len(self.subject_list)

        # Infosource - iterate over the list of contrasts
        infosource_groupanalysis = Node(
            IdentityInterface(
                fields=['subjects', 'model_type'],
                subjects = self.subject_list),
                name='infosource_groupanalysis')
        infosource_groupanalysis.iterables = [('model_type', self.model_list)]

        # SelectFiles
        templates = {
            'contrast' : join(self.directories.output_dir,
                'l1_analysis_{model_type}', '_subject_id_*', 'con_0001.nii'),
                'participants' : join(self.directories.dataset_dir, 'participants.tsv')
            }

        selectfiles_groupanalysis = Node(SelectFiles(
            templates,
            base_directory = self.directories.results_dir,
            force_list= True),
            name="selectfiles_groupanalysis")

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

        # Estimate model
        estimate_model = Node(EstimateModel(
            estimation_method={'Classical':1}),
            name = "estimate_model")

        # Estimate contrasts
        estimate_contrast = Node(EstimateContrast(
            group_contrast=True),
            name = "estimate_contrast")

        # Create thresholded maps
        threshold = MapNode(Threshold(
            use_fwe_correction = True,
            height_threshold_type = 'p-value',
            force_activation = False),
            name = "threshold", iterfield = ['stat_image', 'contrast_index'])

        l2_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = f'l2_analysis_{method}_nsub_{nb_subjects}')

        l2_analysis.connect([
            (infosource_groupanalysis, selectfiles_groupanalysis, [
                ('model_type', 'model_type')]),
            (infosource_groupanalysis, sub_contrasts, [
                ('subjects', 'subject_list')]),
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
            (estimate_model, datasink_groupanalysis, [
                ('mask_image', f"l2_analysis_{method}_nsub_{nb_subjects}.@mask")]),
            (estimate_contrast, datasink_groupanalysis, [
                ('spm_mat_file', f"l2_analysis_{method}_nsub_{nb_subjects}.@spm_mat"),
                ('spmT_images', f"l2_analysis_{method}_nsub_{nb_subjects}.@T"),
                ('con_images', f"l2_analysis_{method}_nsub_{nb_subjects}.@con")]),
            (threshold, datasink_groupanalysis, [
                ('thresholded_map', f"l2_analysis_{method}_nsub_{nb_subjects}.@thresh")])])

        if method in ('equalRange', 'equalIndifference'):
            contrasts = [('Group', 'T', ['mean'], [1]), ('Group', 'T', ['mean'], [-1])]

            # Specify design matrix
            one_sample_t_test_design = Node(OneSampleTTestDesign(),
                name = 'one_sample_t_test_design')

            l2_analysis.connect([
                (sub_contrasts, one_sample_t_test_design, [(f"{method}_files", 'in_files')]),
                (one_sample_t_test_design, estimate_model, [('spm_mat_file', 'spm_mat_file')])])

            threshold.inputs.contrast_index = [1, 2]
            threshold.synchronize = True

        elif method == 'groupComp':
            contrasts = [(
                'Eq range vs Eq indiff in loss', 'T', ['Group_{1}', 'Group_{2}'], [1, -1])]

            # Specify design matrix
            two_sample_t_test_design = Node(TwoSampleTTestDesign(),
                name = 'two_sample_t_test_design')

            l2_analysis.connect([
                (sub_contrasts, two_sample_t_test_design, [
                    ('equalRange_files', "group1_files"),
                    ('equalIndifference_files', 'group2_files')]),
                (two_sample_t_test_design, estimate_model, [("spm_mat_file", "spm_mat_file")])])

            threshold.inputs.contrast_index = [1]
            threshold.synchronize = True

        estimate_contrast.inputs.contrasts = contrasts

        return l2_analysis

    def get_group_level_outputs(self):
        """ Return all names for the files the group level analysis is supposed to generate. """

        # Handle equalRange and equalIndifference
        parameters = {
            'model_type': self.model_list,
            'method': ['equalRange', 'equalIndifference'],
            'file': [
                'con_0001.nii', 'con_0002.nii', 'mask.nii', 'SPM.mat',
                'spmT_0001.nii', 'spmT_0002.nii',
                join('_threshold0', 'spmT_0001_thr.nii'), join('_threshold1', 'spmT_0002_thr.nii')
                ],
            'nb_subjects': [str(len(self.subject_list))]
        }
        parameter_sets = product(*parameters.values())
        template = join(
            self.directories.output_dir,
            'l2_analysis_{method}_nsub_{nb_subjects}',
            '_model_type_{model_type}',
            '{file}'
            )

        return_list = [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

        # Handle groupComp
        parameters = {
            'model_type': ['loss'],
            'method': ['groupComp'],
            'file': [
                'con_0001.nii', 'mask.nii', 'SPM.mat', 'spmT_0001.nii',
                join('_threshold0', 'spmT_0001_thr.nii')
                ],
            'nb_subjects' : [str(len(self.subject_list))]
        }
        parameter_sets = product(*parameters.values())

        return_list += [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

        return return_list

    def get_hypotheses_outputs(self):
        """ Return all hypotheses output file names. """
        nb_sub = len(self.subject_list)
        files = [
            join(f'l2_analysis_equalIndifference_nsub_{nb_sub}', '_model_type_gain', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'l2_analysis_equalIndifference_nsub_{nb_sub}', '_model_type_gain', 'spmT_0001.nii'),
            join(f'l2_analysis_equalRange_nsub_{nb_sub}', '_model_type_gain', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'l2_analysis_equalRange_nsub_{nb_sub}', '_model_type_gain', 'spmT_0001.nii'),
            join(f'l2_analysis_equalIndifference_nsub_{nb_sub}', '_model_type_gain', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'l2_analysis_equalIndifference_nsub_{nb_sub}', '_model_type_gain', 'spmT_0001.nii'),
            join(f'l2_analysis_equalRange_nsub_{nb_sub}', '_model_type_gain', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'l2_analysis_equalRange_nsub_{nb_sub}', '_model_type_gain', 'spmT_0001.nii'),
            join(f'l2_analysis_equalIndifference_nsub_{nb_sub}', '_model_type_loss', '_threshold1', 'spmT_0002_thr.nii'),
            join(f'l2_analysis_equalIndifference_nsub_{nb_sub}', '_model_type_loss', 'spmT_0002.nii'),
            join(f'l2_analysis_equalRange_nsub_{nb_sub}', '_model_type_loss', '_threshold1', 'spmT_0002_thr.nii'),
            join(f'l2_analysis_equalRange_nsub_{nb_sub}', '_model_type_loss', 'spmT_0002.nii'),
            join(f'l2_analysis_equalIndifference_nsub_{nb_sub}', '_model_type_loss', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'l2_analysis_equalIndifference_nsub_{nb_sub}', '_model_type_loss', 'spmT_0001.nii'),
            join(f'l2_analysis_equalRange_nsub_{nb_sub}', '_model_type_loss', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'l2_analysis_equalRange_nsub_{nb_sub}', '_model_type_loss', 'spmT_0001.nii'),
            join(f'l2_analysis_groupComp_nsub_{nb_sub}', '_model_type_loss', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'l2_analysis_groupComp_nsub_{nb_sub}', '_model_type_loss', 'spmT_0001.nii')
        ]
        return [join(self.directories.output_dir, f) for f in files]
