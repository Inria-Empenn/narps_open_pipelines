#!/usr/bin/python
# coding: utf-8

""" Write the work of NARPS team O21U using Nipype """

from os.path import join
from itertools import product

from nipype import Workflow, Node, MapNode
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces.utility import IdentityInterface, Function, Merge
from nipype.interfaces.io import SelectFiles, DataSink, DataGrabber
from nipype.interfaces.fsl.preprocess import SUSAN
from nipype.interfaces.fsl.model import (
    Level1Design, FEATModel, FLAMEO, FILMGLS, MultipleRegressDesign,
    FSLCommand, SmoothEstimate, Cluster
    )
from nipype.interfaces.fsl.utils import ImageMaths, ImageStats, Merge as FSLMerge

from narps_open.utils.configuration import Configuration
from narps_open.pipelines import Pipeline
from narps_open.data.task import TaskInformation
from narps_open.data.participants import get_group
from narps_open.core.common import remove_parent_directory

# Setup FSL
FSLCommand.set_default_output_type('NIFTI_GZ')

class PipelineTeamO21U(Pipeline):
    """ A class that defines the pipeline of team O21U """

    def __init__(self):
        super().__init__()
        self.fwhm = 5.0
        self.team_id = 'O21U'
        self.contrast_list = ['1', '2']
        self.run_level_contrasts = [
            ('effect_of_gain', 'T', ['gain', 'loss'], [1, 0]),
            ('effect_of_loss', 'T', ['gain', 'loss'], [0, 1])
            ]

    def get_preprocessing(self):
        """ Create the preprocessing workflow """

        # Initiate the workflow
        preprocessing = Workflow(
            base_dir = self.directories.working_dir,
            name = 'preprocessing'
            )

        # IdentityInterface Node - Iterate on subject and runs
        information_source = Node(IdentityInterface(
            fields = ['subject_id', 'run_id']),
            name = 'information_source')
        information_source.iterables = [
            ('subject_id', self.subject_list),
            ('run_id', self.run_list)
            ]

        # SelectFiles - Get necessary files
        templates = {
            'func' : join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'),
            'mask' : join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz')
        }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir
        preprocessing.connect(information_source, 'subject_id', select_files, 'subject_id')
        preprocessing.connect(information_source, 'run_id', select_files, 'run_id')

        # ImageMaths - Convert func to float representation
        func_to_float = Node(ImageMaths(), name = 'func_to_float')
        func_to_float.inputs.out_data_type = 'float'
        func_to_float.inputs.op_string = ''
        func_to_float.inputs.suffix = '_dtype'
        preprocessing.connect(select_files, 'func', func_to_float, 'in_file')

        # ImageMaths - Mask the functional image
        mask_func = Node(ImageMaths(), name = 'mask_func')
        mask_func.inputs.suffix = '_thresh'
        mask_func.inputs.op_string = '-mas'
        preprocessing.connect(func_to_float, 'out_file', mask_func, 'in_file')
        preprocessing.connect(select_files, 'mask', mask_func, 'in_file2')

        # ImageMaths - Compute the mean image of each time point
        mean_func = Node(ImageMaths(), name = 'mean_func')
        mean_func.inputs.suffix = '_mean'
        mean_func.inputs.op_string = '-Tmean'
        preprocessing.connect(mask_func, 'out_file', mean_func, 'in_file')

        # ImageStats - Compute the median value of each time point
        # (only because it's needed by SUSAN)
        median_value = Node(ImageStats(), name = 'median_value')
        median_value.inputs.op_string = '-k %s -p 50'
        preprocessing.connect(select_files, 'func', median_value, 'in_file')
        preprocessing.connect(select_files, 'mask', median_value, 'mask_file')

        # Merge - Merge the median values with the mean functional images into a coupled list
        merge_median = Node(Merge(2), name = 'merge_median')
        preprocessing.connect(mean_func, 'out_file', merge_median, 'in1')
        preprocessing.connect(median_value, 'out_stat', merge_median, 'in2')

        # SUSAN - Smoothing funk
        smooth_func = Node(SUSAN(), name = 'smooth_func')
        smooth_func.inputs.fwhm = 4.9996179300001655 # see Chen et. al. 2022

        # Define a function to get the brightness threshold for SUSAN
        get_brightness_threshold = lambda median : 0.75 * median

        # Define a function to get the usans for SUSAN
        get_usans = lambda value : [tuple([value[0], 0.75 * value[1]])]

        preprocessing.connect(mask_func, 'out_file', smooth_func, 'in_file')
        preprocessing.connect(
            median_value, ('out_stat', get_brightness_threshold),
            smooth_func, 'brightness_threshold')
        preprocessing.connect(merge_median, ('out', get_usans), smooth_func, 'usans')

        # TODO : Mask the smoothed data ?
        """
        maskfunc3 = pe.MapNode(
            interface=fsl.ImageMaths(op_string='-mas'),
            name='maskfunc3',
            iterfield=['in_file','in_file2'])
        wf.connect(smooth, 'smoothed_file', maskfunc3, 'in_file')
        wf.connect(dilatemask, 'out_file', maskfunc3, 'in_file2')
        """

        # Define a function to get the scaling factor for intensity normalization
        get_intensity_normalization_scale = lambda median : '-mul %.10f' % (10000. / median)

        # ImageMaths - Scale each time point so that its median value is 10000
        normalize_intensity = Node(ImageMaths(), name = 'normalize_intensity')
        normalize_intensity.inputs.suffix = '_intnorm'
        preprocessing.connect(smooth_func, 'smoothed_file', normalize_intensity, 'in_file')
        preprocessing.connect(
            median_value, ('out_stat', get_intensity_normalization_scale),
            normalize_intensity, 'op_string')

        # ImageMaths - Generate a mean functional image from the scaled data
        mean_func_2 = Node(ImageMaths(), name = 'mean_func_2')
        mean_func_2.inputs.op_string = '-Tmean'
        preprocessing.connect(normalize_intensity, 'out_file', mean_func_2, 'in_file')

        # Function get_high_pass_filter_command - Build command line for temporal highpass filter
        def get_high_pass_filter_command(in_file, repetition_time, high_pass_filter_cutoff):
            """ Create command line for high pass filtering using image maths """
            return f'-bptf {high_pass_filter_cutoff / (2.0 * repetition_time)} -1 -add {in_file}'

        high_pass_command = Node(Function(
            function = get_high_pass_filter_command,
            input_names = ['in_file', 'repetition_time', 'high_pass_filter_cutoff'],
            output_names = ['command']
            ), name = 'high_pass_command')
        high_pass_command.inputs.high_pass_filter_cutoff = 100.0 #seconds
        high_pass_command.inputs.repetition_time = TaskInformation()['RepetitionTime']
        preprocessing.connect(mean_func_2, 'out_file', high_pass_command, 'in_file')

        # ImageMaths - Perform temporal highpass filtering on the data
        high_pass_filter = Node(ImageMaths(), name = 'high_pass_filter')
        high_pass_filter.inputs.suffix = '_tempfilt'
        preprocessing.connect(normalize_intensity, 'out_file', high_pass_filter, 'in_file')
        preprocessing.connect(high_pass_command, 'command', high_pass_filter, 'op_string')

        # DataSink Node - store the wanted results in the wanted repository
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir
        preprocessing.connect(
            high_pass_filter, 'out_file', data_sink, 'preprocessing.@filtered_file')

        # Remove large files, if requested
        if Configuration()['pipelines']['remove_unused_data']:

            # Merge Node - Merge func file names to be removed after datasink node is performed
            merge_removable_files = Node(Merge(6), name = 'merge_removable_files')
            merge_removable_files.inputs.ravel_inputs = True
            preprocessing.connect(func_to_float, 'out_file', merge_removable_files, 'in1')
            preprocessing.connect(mask_func, 'out_file', merge_removable_files, 'in2')
            preprocessing.connect(mean_func, 'out_file', merge_removable_files, 'in3')
            preprocessing.connect(smooth_func, 'smoothed_file', merge_removable_files, 'in4')
            preprocessing.connect(normalize_intensity, 'out_file', merge_removable_files, 'in5')
            preprocessing.connect(high_pass_filter, 'out_file', merge_removable_files, 'in6')

            # Function Nodes remove_files - Remove sizeable func files once they aren't needed
            remove_dirs = MapNode(Function(
                function = remove_parent_directory,
                input_names = ['_', 'file_name'],
                output_names = []
                ), name = 'remove_dirs', iterfield = 'file_name')
            preprocessing.connect(merge_removable_files, 'out', remove_dirs, 'file_name')
            preprocessing.connect(data_sink, 'out_file', remove_dirs, '_')

        return preprocessing

    def get_preprocessing_outputs(self):
        """ Return the names of the files the preprocessing is supposed to generate. """

        # Outputs from preprocessing (intensity normalized files)
        parameters = {
            'subject_id': self.subject_list,
            'run_id': self.run_list,
        }
        parameter_sets = product(*parameters.values())
        template = join(
            self.directories.output_dir, 'preprocessing',
            '_run_id_{run_id}_subject_id_{subject_id}',
            'sub-{subject_id}_task-MGT_run-{run_id}_bold_space-MNI152NLin2009cAsym_preproc_dtype_thresh_smooth_intnorm_tempfilt.nii.gz')

        return [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

    def get_subject_information(event_file):
        """
        Create Bunchs for specifyModel.

        Parameters :
        - event_file : str, file corresponding to the run and the subject to analyze

        Returns :
        - subject_info : list of Bunch for 1st level analysis.
        """
        from nipype.interfaces.base import Bunch

        onsets = []
        durations = []
        amplitudes_trial = []
        amplitudes_gain = []
        amplitudes_loss = []

        with open(event_file, 'rt') as file:
            next(file)  # skip the header

            for line in file:
                info = line.strip().split()
                onsets.append(float(info[0]))
                durations.append(float(info[1]))
                amplitudes_trial.append(1.0)
                amplitudes_gain.append(float(info[2]))
                amplitudes_loss.append(float(info[3]))

        # ... included a model derived nuisance variable that entered the analysis along
        # with the two main independent variables (germane to the hypotheses evaluated)
        # of gain and loss for that trial. We also included an intercept term.
        # These variables were modeled with a duration of 4 seconds and included their temporal derivatives.
        # TODO :intercept and nuisance variable

        return [
            Bunch(
                conditions = ['trial', 'gain', 'loss'],
                onsets = [onsets] * 3,
                durations = [durations] * 3 ,
                amplitudes = [amplitudes_trial, amplitudes_gain, amplitudes_loss],
                )
            ]

    def get_confounds_file(in_file):
        """
        Create a tsv file with only desired confounds per subject per run.

        Parameters :
        - in_file : path to the subject confounds file (i.e. one per run)

        Return :
        - confounds_file : paths to new files containing only desired confounds.
        """
        from os.path import abspath

        from pandas import read_csv, DataFrame
        from numpy import array, transpose

        # Extract confounds from the fMRIPrep file
        data_frame = read_csv(in_file, sep = '\t', header=0)
        retained_confounds = DataFrame(transpose(array([
            data_frame['FramewiseDisplacement'], data_frame['X'], data_frame['Y'], data_frame['Z'],
            data_frame['RotX'], data_frame['RotY'], data_frame['RotZ']
        ])))

        # Write confounds to a file
        confounds_file = abspath('confounds_file.tsv')
        with open(confounds_file, 'w', encoding = 'utf-8') as writer:
            writer.write(retained_confounds.to_csv(
                sep = '\t', index = False, header = False, na_rep = '0.0'))

        return confounds_file

    def get_run_level_analysis(self):
        """
        Create the run level analysis workflow.

        Returns:
            - run_level : nipype.WorkFlow
        """
        # Create run level analysis workflow
        run_level = Workflow(
            base_dir = self.directories.working_dir,
            name = 'run_level_analysis'
            )

        # IdentityInterface Node - Iterate on subject and runs
        information_source = Node(IdentityInterface(
            fields = ['subject_id', 'run_id']),
            name = 'information_source')
        information_source.iterables = [
            ('subject_id', self.subject_list),
            ('run_id', self.run_list)
            ]

        # SelectFiles - Get necessary files
        templates = {
            'func' : join(self.directories.output_dir,
                'preprocessing', '_run_id_{run_id}_subject_id_{subject_id}',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_space-MNI152NLin2009cAsym_preproc_dtype_thresh_smooth_intnorm_tempfilt.nii.gz'),
            'mask' : join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz'),
            'events' : join('sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_events.tsv'),
            'confounds' : join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_confounds.tsv')
        }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir
        run_level.connect(information_source, 'subject_id', select_files, 'subject_id')
        run_level.connect(information_source, 'run_id', select_files, 'run_id')

        # Get Subject Info - get subject specific condition information
        subject_information = Node(Function(
            function = self.get_subject_information,
            input_names = ['event_file'],
            output_names = ['subject_info']
            ), name = 'subject_information')
        run_level.connect(select_files, 'events', subject_information, 'event_file')

        # Get Confounds - get confounds
        select_confounds = Node(Function(
            function = self.get_confounds_file,
            input_names = ['in_file'],
            output_names = ['confounds_file']
            ), name = 'select_confounds', iterfield = ['run_id'])
        run_level.connect(select_files, 'confounds', select_confounds, 'in_file')

        # SpecifyModel Node - Generate run level model
        specify_model = Node(SpecifyModel(), name = 'specify_model')
        specify_model.inputs.high_pass_filter_cutoff = 100
        specify_model.inputs.input_units = 'secs'
        specify_model.inputs.time_repetition = TaskInformation()['RepetitionTime']
        run_level.connect(select_files, 'func', specify_model, 'functional_runs')
        run_level.connect(subject_information, 'subject_info', specify_model, 'subject_info')
        run_level.connect(select_confounds, 'confounds_file', specify_model, 'realignment_parameters')

        # Level1Design Node - Generate files for run level computation
        model_design = Node(Level1Design(), name = 'model_design')
        model_design.inputs.bases = {'dgamma' : {'derivs' : True }}
        model_design.inputs.interscan_interval = TaskInformation()['RepetitionTime']
        model_design.inputs.model_serial_correlations = True
        model_design.inputs.contrasts = self.run_level_contrasts
        run_level.connect(specify_model, 'session_info', model_design, 'session_info')

        # FEATModel Node - Generate run level model
        model_generation = Node(FEATModel(), name = 'model_generation')
        run_level.connect(model_design, 'ev_files', model_generation, 'ev_files')
        run_level.connect(model_design, 'fsf_files', model_generation, 'fsf_file')

        # FILMGLS Node - Estimate first level model
        model_estimate = Node(FILMGLS(), name='model_estimate')
        model_estimate.inputs.smooth_autocorr = True
        model_estimate.inputs.mask_size = 5
        model_estimate.inputs.threshold = 1000
        run_level.connect(select_files, 'func', model_estimate, 'in_file')
        run_level.connect(model_generation, 'con_file', model_estimate, 'tcon_file')
        run_level.connect(model_generation, 'design_file', model_estimate, 'design_file')

        # DataSink Node - store the wanted results in the wanted directory
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir
        run_level.connect(model_estimate, 'results_dir', data_sink, 'run_level_analysis.@results')

        return run_level

    def get_run_level_outputs(self):
        """ Return the names of the files the run level analysis is supposed to generate. """

        parameters = {
            'run_id' : self.run_list,
            'subject_id' : self.subject_list,
            'contrast_id' : self.contrast_list,
        }
        parameter_sets = product(*parameters.values())
        output_dir = join(self.directories.output_dir,
            'run_level_analysis', '_run_id_{run_id}_subject_id_{subject_id}')
        templates = [
                join(output_dir, 'results', 'cope{contrast_id}.nii.gz'),
                join(output_dir, 'results', 'tstat{contrast_id}.nii.gz'),
                join(output_dir, 'results', 'varcope{contrast_id}.nii.gz'),
                join(output_dir, 'results', 'zstat{contrast_id}.nii.gz')
            ]
        return [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets for template in templates]

    def get_subject_level_contrasts(subject_list: list, run_list: list):
        """ Return a list of contrasts and a dictionary of regressors
            for the subject level analysis """
        nb_subjects = len(subject_list)
        nb_runs = len(run_list)

        # Create a list of EVs (explanatory variables), one per subject
        ev_list = [f'ev{s}' for s in range(1,nb_subjects+1)]

        # Create a list of contrasts
        contrasts = [['', 'T', ev_list, [0.0] * len(ev_list)] for s in range(nb_subjects)]
        for index, _ in enumerate(contrasts):
            contrasts[index][3][index] = 1.0

        # Create a regressors dict
        regressors = {e:[0.0] * nb_subjects * nb_runs for e in ev_list}
        for index, key in enumerate(regressors.keys()):
            regressors[key][index * nb_runs:index * nb_runs + nb_runs] = [1.0] * nb_runs

        return contrasts, regressors

    def get_subject_level_analysis(self):
        """
        Create the subject level analysis workflow.

        Returns:
        - subject_level_analysis : nipype.WorkFlow
        """
        # Second level analysis workflow.
        subject_level = Workflow(
            base_dir = self.directories.working_dir,
            name = 'subject_level_analysis')

        # Infosource Node - To iterate on contrasts
        information_source = Node(IdentityInterface(
            fields = ['contrast_id']),
            name = 'information_source')
        information_source.iterables = [('contrast_id', self.contrast_list)]

        # Datagrabber - Select copes and varcopes from the run level analysis
        select_copes = Node(DataGrabber(
            infields = ['contrast_id'],
            outfields = ['copes', 'varcopes']
            ), name = 'select_copes')
        select_copes.inputs.base_directory = join(
            self.directories.output_dir, 'run_level_analysis')
        select_copes.inputs.template = '*'
        select_copes.inputs.field_template = {
            'copes': join('_run_id_%s_subject_id_%s', 'results', 'cope%s.nii.gz'),
            'varcopes': join('_run_id_%s_subject_id_%s', 'results', 'varcope%s.nii.gz'),
            }
        select_copes.inputs.template_args = {
            'copes': [[r,s,'contrast_id'] for s,r in product(self.subject_list, self.run_list)],
            'varcopes': [[r,s,'contrast_id'] for s,r in product(self.subject_list, self.run_list)]
            }
        select_copes.inputs.sort_filelist = False
        subject_level.connect(information_source, 'contrast_id', select_copes, 'contrast_id')

        # Datagrabber - Select brain masks
        select_masks = Node(DataGrabber(
            outfields = ['masks']
            ), name = 'select_masks')
        select_masks.inputs.base_directory = join(
            self.directories.dataset_dir, 'derivatives', 'fmriprep')
        select_masks.inputs.template = '*_brainmask.nii.gz'
        select_masks.inputs.field_template = {
            'masks': join('sub-%s', 'func',
                'sub-%s_task-MGT_run-%s_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz')
            }
        select_masks.inputs.template_args = {
            'masks': [[s, s, r] for s, r in product(self.subject_list, self.run_list)]
            }
        select_masks.inputs.sort_filelist = False

        # Merge - Merge copes files for each subject
        merge_copes = Node(FSLMerge(), name = 'merge_copes')
        merge_copes.inputs.dimension = 't'
        subject_level.connect(select_copes, 'copes', merge_copes, 'in_files')

        # Merge - Merge varcopes files for all runs for each subject
        merge_varcopes = Node(FSLMerge(), name = 'merge_varcopes')
        merge_varcopes.inputs.dimension = 't'
        subject_level.connect(select_copes, 'varcopes', merge_varcopes, 'in_files')

        # Merge - Merge mask files for all runs for each subject
        merge_masks = Node(FSLMerge(), name = 'merge_masks')
        merge_masks.inputs.dimension = 't'
        subject_level.connect(select_masks, 'masks', merge_masks, 'in_files')

        # ImageMaths - Create a mask by selecting the min value along time axis
        mask_min = Node(ImageMaths(), name = 'mask_min')
        mask_min.inputs.op_string = '-Tmin'
        subject_level.connect(merge_masks, 'merged_file', mask_min, 'in_file')

        # ImageMaths - Mask copes
        mask_copes = Node(ImageMaths(), name = 'mask_copes')
        mask_copes.inputs.op_string = '-mas'
        subject_level.connect(merge_copes, 'merged_file', mask_copes, 'in_file')
        subject_level.connect(mask_min, 'out_file', mask_copes, 'in_file2')

        # ImageMaths - Mask copes
        mask_varcopes = Node(ImageMaths(), name = 'mask_varcopes')
        mask_varcopes.inputs.op_string = '-mas'
        subject_level.connect(merge_varcopes, 'merged_file', mask_varcopes, 'in_file')
        subject_level.connect(mask_min, 'out_file', mask_varcopes, 'in_file2')

        # Function - Create contrasts and regressors for second level
        get_contrasts = Node(Function(
            function = self.get_subject_level_contrasts,
            input_names = ['subject_list', 'run_list'],
            output_names = ['contrasts', 'regressors'],
            ), name='get_contrasts')
        get_contrasts.inputs.subject_list = self.subject_list
        get_contrasts.inputs.run_list = self.run_list

        # MultipleRegressDesign - Model subject level with multiple predictors (subjects)
        generate_model = Node(MultipleRegressDesign(), name = 'generate_model')
        subject_level.connect(get_contrasts, 'contrasts', generate_model, 'contrasts')
        subject_level.connect(get_contrasts, 'regressors', generate_model, 'regressors')

        # FLAMEO Node - Estimate model
        estimate_model = Node(FLAMEO(), name = 'estimate_model')
        estimate_model.inputs.run_mode = 'fe'
        subject_level.connect(mask_min, 'out_file', estimate_model,  'mask_file')
        subject_level.connect(mask_copes, 'out_file', estimate_model, 'cope_file')
        subject_level.connect(mask_varcopes, 'out_file', estimate_model, 'var_cope_file')
        subject_level.connect(generate_model, 'design_mat', estimate_model, 'design_file')
        subject_level.connect(generate_model, 'design_con', estimate_model, 't_con_file')
        subject_level.connect(generate_model, 'design_grp', estimate_model, 'cov_split_file')

        # DataSink Node - store the wanted results in the wanted directory
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir
        subject_level.connect(
            mask_min, 'out_file', data_sink, 'subject_level_analysis.@mask')
        subject_level.connect(estimate_model, 'zstats', data_sink, 'subject_level_analysis.@stats')
        subject_level.connect(
            estimate_model, 'tstats', data_sink, 'subject_level_analysis.@tstats')
        subject_level.connect(estimate_model, 'copes', data_sink, 'subject_level_analysis.@copes')
        subject_level.connect(
            estimate_model, 'var_copes', data_sink, 'subject_level_analysis.@varcopes')

        return subject_level

    def get_subject_level_outputs(self):
        """ Return the names of the files the subject level analysis is supposed to generate. """

        # Copes, varcopes, stats
        parameters = {
            'contrast_id' : self.contrast_list,
            'subject_ev' : range(1, 1+len(self.subject_list))
        }
        parameter_sets = product(*parameters.values())
        output_dir = join(self.directories.output_dir, 'subject_level_analysis')
        templates = [
            join(output_dir, '_contrast_id_{contrast_id}', 'cope{subject_ev}.nii.gz'),
            join(output_dir, '_contrast_id_{contrast_id}', 'tstat{subject_ev}.nii.gz'),
            join(output_dir, '_contrast_id_{contrast_id}', 'varcope{subject_ev}.nii.gz'),
            join(output_dir, '_contrast_id_{contrast_id}', 'zstat{subject_ev}.nii.gz')
            ]
        return_list = [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets for template in templates]

        # Mask
        return_list.append(join(output_dir,
            f'sub-{self.subject_list[0]}_task-MGT_run-{self.run_list[0]}_bold_space-MNI152NLin2009cAsym_brainmask_merged_maths.nii.gz'
            ))

        return return_list

    def get_one_sample_t_test_regressors(subject_list: list) -> dict:
        """
        Create dictionary of regressors for one sample t-test group analysis.

        Parameters:
            - subject_list: ids of subject in the group for which to do the analysis

        Returns:
            - dict containing named lists of regressors.
        """

        return dict(group_mean = [1 for _ in subject_list])

    def get_two_sample_t_test_regressors(
        subject_list: list,
        equal_range_ids: list,
        equal_indifference_ids: list
        ) -> dict:
        """
        Create dictionary of regressors for two sample t-test group analysis.

        Parameters:
            - equal_range_ids: ids of subjects in equal range group
            - equal_indifference_ids: ids of subjects in equal indifference group
            - subject_list: ids of subject for which to do the analysis

        Returns:
            - regressors, dict: containing named lists of regressors.
            - groups, list: group identifiers to distinguish groups in FSL analysis.
        """

        # Create 2 lists containing n_sub values which are
        #  * 1 if the participant is on the group
        #  * 0 otherwise
        equal_range_regressors = [1 if i in equal_range_ids else 0 for i in subject_list]
        equal_indifference_regressors = [
            1 if i in equal_indifference_ids else 0 for i in subject_list
            ]

        # Create regressors output : a dict with the two list
        regressors = dict(
            equalRange = equal_range_regressors,
            equalIndifference = equal_indifference_regressors
        )

        # Create groups outputs : a list with 1 for equalRange subjects and 2 for equalIndifference
        groups = [1 if i == 1 else 2 for i in equal_range_regressors]

        return regressors, groups

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
            - group_level: nipype.WorkFlow
        """
        # Compute the number of participants used to do the analysis
        nb_subjects = len(self.subject_list)

        # Create subject lists & convert subject ids to EVs ids used at subject_level
        equal_range_subjects = [s for s in get_group('equalRange') if s in self.subject_list]
        equal_range_evs = [self.subject_list.index(s)+1 for s in equal_range_subjects]
        equal_indifference_subjects = [
            s for s in get_group('equalIndifference') if s in self.subject_list]
        equal_indifference_evs = [
            self.subject_list.index(s)+1 for s in equal_indifference_subjects]

        selected_subjects = []
        selected_evs = []
        if method == 'equalRange':
            selected_subjects = equal_range_subjects
            selected_evs = equal_range_evs
        elif method == 'equalIndifference':
            selected_subjects = equal_indifference_subjects
            selected_evs = equal_indifference_evs
        else:
            selected_subjects = equal_range_subjects + equal_indifference_subjects
            selected_evs = equal_range_evs + equal_indifference_evs

        # Declare the workflow
        group_level = Workflow(
            base_dir = self.directories.working_dir,
            name = f'group_level_analysis_{method}_nsub_{nb_subjects}')

        # Infosource Node - Iterate over run level contrasts
        information_source = Node(IdentityInterface(
            fields = ['contrast_id']),
            name = 'information_source')
        information_source.iterables = [('contrast_id', self.contrast_list)]

        # Datagrabber - Select copes and varcopes from the run level analysis
        select_copes = Node(DataGrabber(
            infields = ['contrast_id'],
            outfields = ['copes', 'varcopes']
            ), name = 'select_copes')
        select_copes.inputs.base_directory = join(
            self.directories.output_dir, 'subject_level_analysis')
        select_copes.inputs.template = '*'
        select_copes.inputs.field_template = {
            'copes': join('_contrast_id_%s', 'cope%s.nii.gz'),
            'varcopes': join('_contrast_id_%s', 'varcope%s.nii.gz')
            }
        select_copes.inputs.template_args = {
            'copes': [['contrast_id', s] for s in selected_evs],
            'varcopes': [['contrast_id', s] for s in selected_evs]
            }
        select_copes.inputs.sort_filelist = False
        group_level.connect(information_source, 'contrast_id', select_copes, 'contrast_id')

        # Datagrabber - Select brain masks
        select_masks = Node(DataGrabber(
            outfields = ['masks']
            ), name = 'select_masks')
        select_masks.inputs.base_directory = join(
            self.directories.dataset_dir, 'derivatives', 'fmriprep')
        select_masks.inputs.template = '*_brainmask.nii.gz'
        select_masks.inputs.field_template = {
            'masks': join('sub-%s', 'func',
                'sub-%s_task-MGT_run-%s_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz')
            }
        select_masks.inputs.template_args = {
            'masks': [[s, s, r] for s, r in product(selected_subjects, self.run_list)]
            }
        select_masks.inputs.sort_filelist = False

        # Merge - Merge copes files for each subject
        merge_copes = Node(FSLMerge(), name = 'merge_copes')
        merge_copes.inputs.dimension = 't'
        group_level.connect(select_copes, 'copes', merge_copes, 'in_files')

        # Merge - Merge varcopes files for all runs for each subject
        merge_varcopes = Node(FSLMerge(), name = 'merge_varcopes')
        merge_varcopes.inputs.dimension = 't'
        group_level.connect(select_copes, 'varcopes', merge_varcopes, 'in_files')

        # Merge - Merge mask files for all runs for each subject
        merge_masks = Node(FSLMerge(), name = 'merge_masks')
        merge_masks.inputs.dimension = 't'
        group_level.connect(select_masks, 'masks', merge_masks, 'in_files')

        # ImageMaths - Create a mask by selecting the min value along time axis
        mask_min = Node(ImageMaths(), name = 'mask_min')
        mask_min.inputs.op_string = '-Tmin'
        group_level.connect(merge_masks, 'merged_file', mask_min, 'in_file')

        # ImageMaths - Mask copes
        mask_copes = Node(ImageMaths(), name = 'mask_copes')
        mask_copes.inputs.op_string = '-mas'
        group_level.connect(merge_copes, 'merged_file', mask_copes, 'in_file')
        group_level.connect(mask_min, 'out_file', mask_copes, 'in_file2')

        # ImageMaths - Mask copes
        mask_varcopes = Node(ImageMaths(), name = 'mask_varcopes')
        mask_varcopes.inputs.op_string = '-mas'
        group_level.connect(merge_varcopes, 'merged_file', mask_varcopes, 'in_file')
        group_level.connect(mask_min, 'out_file', mask_varcopes, 'in_file2')

        # MultipleRegressDesign Node - Specify model
        specify_model = Node(MultipleRegressDesign(), name = 'specify_model')

        # Setup model contrats and regressors
        if method in ('equalIndifference', 'equalRange'):
            # One sample t-test
            specify_model.inputs.contrasts = [
                ['group_mean', 'T', ['group_mean'], [1]],
                ['group_mean_neg', 'T', ['group_mean'], [-1]]
                ]

            # Function Node get_one_sample_t_test_regressors
            regressors_one_sample = Node(
                Function(
                    function = self.get_one_sample_t_test_regressors,
                    input_names = ['subject_list'],
                    output_names = ['regressors']
                ),
                name = 'regressors_one_sample',
            )
            regressors_one_sample.inputs.subject_list = selected_subjects
            group_level.connect(regressors_one_sample, 'regressors', specify_model, 'regressors')

        elif method == 'groupComp':
            # Two sample t-test
            specify_model.inputs.contrasts = [
                ['equalRange_sup', 'T', ['equalRange', 'equalIndifference'], [1, -1]]
            ]

            # Function Node get_two_sample_t_test_regressors
            regressors_two_sample = Node(
                Function(
                    function = self.get_two_sample_t_test_regressors,
                    input_names = [
                        'subject_list',
                        'equal_range_ids',
                        'equal_indifference_ids',
                    ],
                    output_names = ['regressors', 'groups']
                ),
                name = 'regressors_two_sample',
            )
            regressors_two_sample.inputs.subject_list = selected_subjects
            regressors_two_sample.inputs.equal_range_ids = equal_range_subjects
            regressors_two_sample.inputs.equal_indifference_ids = equal_indifference_subjects
            group_level.connect(regressors_two_sample, 'regressors', specify_model, 'regressors')
            group_level.connect(regressors_two_sample, 'groups', specify_model, 'groups')

        # FLAMEO Node - Estimate model
        estimate_model = Node(FLAMEO(), name = 'estimate_model')
        estimate_model.inputs.run_mode = 'flame1'
        estimate_model.inputs.infer_outliers = True
        group_level.connect(mask_min, 'out_file', estimate_model, 'mask_file')
        group_level.connect(merge_copes, 'merged_file', estimate_model, 'cope_file')
        group_level.connect(merge_varcopes, 'merged_file', estimate_model, 'var_cope_file')
        group_level.connect(specify_model, 'design_mat', estimate_model, 'design_file')
        group_level.connect(specify_model, 'design_con', estimate_model, 't_con_file')
        group_level.connect(specify_model, 'design_grp', estimate_model, 'cov_split_file')

        # SmoothEstimate - Smoothness estimation to get dlh and volume for thresholding
        smooth_estimate = Node(SmoothEstimate(), name = 'smooth_estimate')
        smooth_estimate.inputs.dof = 25
        group_level.connect(mask_min, 'out_file', smooth_estimate, 'mask_file')
        group_level.connect(estimate_model, 'res4d', smooth_estimate, 'residual_fit_file')

        # ImageMaths - Mask zstat files
        mask_zstat = MapNode(
            ImageMaths(),
            iterfield = ['in_file'],
            name = 'mask_zstat'
            )
        mask_zstat.inputs.op_string = '-mas'
        group_level.connect(estimate_model, 'zstats', mask_zstat, 'in_file')
        group_level.connect(mask_min, 'out_file', mask_zstat, 'in_file2')

        # Cluster Node - Perform clustering on statistical output
        cluster = MapNode(
            Cluster(),
            name = 'cluster',
            iterfield = ['in_file', 'cope_file'],
            synchronize = True
            )
        cluster.inputs.threshold = 2.3
        cluster.inputs.pthreshold = 0.05
        cluster.inputs.out_threshold_file = True
        group_level.connect(mask_zstat, 'out_file', cluster, 'in_file')
        group_level.connect(estimate_model, 'copes', cluster, 'cope_file')
        group_level.connect(smooth_estimate, 'dlh', cluster, 'dlh')
        group_level.connect(smooth_estimate, 'volume', cluster, 'volume')

        # Datasink Node - Save important files
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir
        group_level.connect(estimate_model, 'zstats', data_sink,
            f'group_level_analysis_{method}_nsub_{nb_subjects}.@zstats')
        group_level.connect(estimate_model, 'tstats', data_sink,
            f'group_level_analysis_{method}_nsub_{nb_subjects}.@tstats')
        group_level.connect(cluster,'threshold_file', data_sink,
            f'group_level_analysis_{method}_nsub_{nb_subjects}.@threshold_file')

        return group_level

    def get_group_level_outputs(self):
        """ Return all names for the files the group level analysis is supposed to generate. """

        # Handle equalRange and equalIndifference
        parameters = {
            'contrast_id': self.contrast_list,
            'method': ['equalRange', 'equalIndifference'],
            'file': [
                '_cluster0/zstat1_maths_threshold.nii.gz',
                '_cluster1/zstat2_maths_threshold.nii.gz',
                'tstat1.nii.gz',
                'tstat2.nii.gz',
                'zstat1.nii.gz',
                'zstat2.nii.gz'
                ]
        }
        parameter_sets = product(*parameters.values())
        template = join(
            self.directories.output_dir,
            'group_level_analysis_{method}_nsub_'+f'{len(self.subject_list)}',
            '_contrast_id_{contrast_id}',
            '{file}'
            )
        return_list = [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

        # Handle groupComp
        parameters = {
            'contrast_id': self.contrast_list,
            'file': [
                '_cluster0/zstat1_maths_threshold.nii.gz',
                'tstat1.nii.gz',
                'zstat1.nii.gz'
                ]
        }
        parameter_sets = product(*parameters.values())
        template = join(
            self.directories.output_dir,
            f'group_level_analysis_groupComp_nsub_{len(self.subject_list)}',
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
                '_contrast_id_1', 'randomise_tfce_corrp_tstat2'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_1', 'zstat1.nii.gz'),
            # Hypothesis 2
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_1', 'randomise_tfce_corrp_tstat2'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_1', 'zstat1.nii.gz'),
            # Hypothesis 3
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_1', 'randomise_tfce_corrp_tstat2'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_1', 'zstat1.nii.gz'),
            # Hypothesis 4
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_1', 'randomise_tfce_corrp_tstat2'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_1', 'zstat1.nii.gz'),
            # Hypothesis 5
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_2', 'randomise_tfce_corrp_tstat1'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_2', 'zstat2.nii.gz'),
            # Hypothesis 6
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_2', 'randomise_tfce_corrp_tstat1'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_2', 'zstat2.nii.gz'),
            # Hypothesis 7
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_2', 'randomise_tfce_corrp_tstat1'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_2', 'zstat1.nii.gz'),
            # Hypothesis 8
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_2', 'randomise_tfce_corrp_tstat1'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_2', 'zstat1.nii.gz'),
            # Hypothesis 9
            join(f'group_level_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_2', 'randomise_tfce_corrp_tstat1'),
            join(f'group_level_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_2', 'zstat1.nii.gz')
        ]
        return [join(self.directories.output_dir, f) for f in files]
