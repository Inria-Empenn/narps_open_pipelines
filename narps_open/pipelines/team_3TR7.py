#!/usr/bin/python
# coding: utf-8

""" Write the work of NARPS' team 3TR7 using Nipype """

from os.path import join
from itertools import product

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
from narps_open.data.task import TaskInformation
from narps_open.data.participants import get_group
from narps_open.core.interfaces import InterfaceFactory
from narps_open.core.common import list_intersection, elements_in_string, clean_list
from narps_open.utils.configuration import Configuration

class PipelineTeam3TR7(Pipeline):
    """ A class that defines the pipeline of team 3TR7. """

    def __init__(self):
        super().__init__()
        self.fwhm = 8.0
        self.team_id = '3TR7'
        self.contrast_list = ['0001', '0002']
        self.subject_level_contrasts = [
            ('effect_of_gain', 'T', ['gamble', 'gamblexgain^1', 'gamblexloss^1'], [0, 1, 0]),
            ('effect_of_loss', 'T', ['gamble', 'gamblexgain^1', 'gamblexloss^1'], [0, 0, 1])
        ]

    def get_preprocessing(self):
        """ No preprocessing has been done by team 3TR7 """
        return None

    def get_run_level_analysis(self):
        """ No run level analysis has been done by team 3TR7 """
        return None

    # @staticmethod # Starting python 3.10, staticmethod should be used here
    # Otherwise it produces a TypeError: 'staticmethod' object is not callable
    def get_subject_information(event_files: list):
        """ Create Bunchs for SpecifySPMModel.

        Parameters :
        - event_files: list of str, list of events files (one per run) for the subject

        Returns :
        - subject_info : list of Bunch for 1st level analysis.
        """

        from nipype.interfaces.base import Bunch

        onsets = {}
        durations = {}
        weights_gain = {}
        weights_loss = {}

        subject_info = []

        for run_id, event_file in enumerate(event_files):

            trial_key = f'gamble_run{run_id + 1}'
            gain_key = f'gain_run{run_id + 1}'
            loss_key = f'loss_run{run_id + 1}'

            onsets.update({trial_key: []})
            durations.update({trial_key: []})
            weights_gain.update({gain_key: []})
            weights_loss.update({loss_key: []})

            with open(event_file, 'rt') as file:
                next(file)  # skip the header

                for line in file:
                    info = line.strip().split()
                    onsets[trial_key].append(float(info[0]))
                    durations[trial_key].append(float(info[1]))
                    weights_gain[gain_key].append(float(info[2]))
                    weights_loss[loss_key].append(float(info[3]))

            # Create a Bunch per run, i.e. cond1_run1, cond2_run1, etc.
            subject_info.append(
                Bunch(
                    conditions = ['gamble'],
                    onsets = [onsets[trial_key]],
                    durations = [durations[trial_key]],
                    amplitudes = None,
                    tmod = None,
                    pmod = [Bunch(
                        name = ['gain', 'loss'],
                        poly = [1, 1],
                        param = [weights_gain[gain_key], weights_loss[loss_key]]
                        )],
                    regressor_names = None,
                    regressors = None
                ))

        return subject_info

    # @staticmethod # Starting python 3.10, staticmethod should be used here
    # Otherwise it produces a TypeError: 'staticmethod' object is not callable
    def get_confounds_file(filepath, subject_id, run_id, working_dir):
        """
        Create a new tsv files with only desired confounds per subject per run.
        Also computes the first derivative of the motion parameters.

        Parameters :
        - filepath : path to the subject confounds file
        - subject_id : related subject id
        - run_id : related run id
        - working_dir: str, name of the directory for intermediate results

        Return :
        - confounds_file : path to new file containing only desired confounds
        """
        from os import makedirs
        from os.path import join

        from pandas import DataFrame, read_csv
        from numpy import array, transpose

        # Open original confounds file
        data_frame = read_csv(filepath, sep = '\t', header=0)

        # Extract confounds we want to use for the model
        retained_parameters = DataFrame(transpose(array([
            data_frame['X'], data_frame['Y'], data_frame['Z'],
            data_frame['RotX'], data_frame['RotY'], data_frame['RotZ']
            ])))

        # Write confounds to a file
        confounds_file = join(working_dir, 'confounds_files',
            f'confounds_file_sub-{subject_id}_run-{run_id}.tsv')

        makedirs(join(working_dir, 'confounds_files'), exist_ok = True)

        with open(confounds_file, 'w', encoding = 'utf-8') as writer:
            writer.write(retained_parameters.to_csv(
                sep = '\t', index = False, header = False, na_rep = '0.0'))

        return confounds_file

    def get_subject_level_analysis(self):
        """
        Create the subject level analysis workflow.

        Returns:
            - subject_level : nipype.WorkFlow
        """
        # Initialize preprocessing workflow to connect nodes along the way
        subject_level = Workflow(
            base_dir = self.directories.working_dir, name = 'subject_level'
            )

        # Identity interface Node - to iterate over subject_id and run
        info_source = Node(
            IdentityInterface(fields = ['subject_id']),
            name = 'info_source')
        info_source.iterables = [('subject_id', self.subject_list)]

        # Select files from derivatives
        templates = {
            'func': join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-*_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'),
            'confounds' : join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-*_bold_confounds.tsv'),
            'events': join('sub-{subject_id}', 'func',
               'sub-{subject_id}_task-MGT_run-*_events.tsv')
        }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir
        select_files.inputs.sort_filelist = True
        subject_level.connect(info_source, 'subject_id', select_files, 'subject_id')

        # Gunzip - gunzip files because SPM do not use .nii.gz files
        gunzip = MapNode(Gunzip(), name = 'gunzip', iterfield=['in_file'])
        subject_level.connect(select_files, 'func', gunzip, 'in_file')

        # Function node get_subject_info - get subject specific condition information
        subject_info = Node(Function(
            function = self.get_subject_information,
            input_names = ['event_files'],
            output_names = ['subject_info']
            ),
            name = 'subject_info')
        subject_level.connect(select_files, 'events', subject_info, 'event_files')

        # Function node get_confounds_file - Generate confounds files
        confounds = MapNode(Function(
            function = self.get_confounds_file,
            input_names = ['filepath', 'subject_id', 'run_id', 'working_dir'],
            output_names = ['confounds_file']),
            name = 'confounds', iterfield = ['filepath', 'run_id'])
        confounds.inputs.working_dir = self.directories.working_dir
        confounds.inputs.run_id = self.run_list
        subject_level.connect(info_source, 'subject_id', confounds, 'subject_id')
        subject_level.connect(select_files, 'confounds', confounds, 'filepath')

        specify_model = Node(SpecifySPMModel(), name = 'specify_model')
        specify_model.inputs.concatenate_runs = False
        specify_model.inputs.input_units = 'secs'
        specify_model.inputs.output_units = 'secs'
        specify_model.inputs.time_repetition = TaskInformation()['RepetitionTime']
        specify_model.inputs.high_pass_filter_cutoff = 128
        specify_model.overwrite = False
        subject_level.connect(subject_info, 'subject_info', specify_model, 'subject_info')
        subject_level.connect(confounds, 'confounds_file', specify_model, 'realignment_parameters')
        subject_level.connect(gunzip, 'out_file', specify_model, 'functional_runs')

        model_design = Node(Level1Design(), name = 'model_design')
        model_design.inputs.bases = {'hrf': {'derivs': [0, 0]}}
        model_design.inputs.timing_units = 'secs'
        model_design.inputs.interscan_interval = TaskInformation()['RepetitionTime']
        model_design.overwrite = False
        subject_level.connect(specify_model, 'session_info', model_design, 'session_info')

        model_estimate = Node(EstimateModel(), name = 'model_estimate')
        model_estimate.inputs.estimation_method = {'Classical': 1}
        model_estimate.overwrite = False
        subject_level.connect(model_design, 'spm_mat_file', model_estimate, 'spm_mat_file')

        contrast_estimate = Node(EstimateContrast(), name = 'contraste_estimate')
        contrast_estimate.inputs.contrasts = self.subject_level_contrasts
        contrast_estimate.config = {'execution': {'remove_unnecessary_outputs': False}}
        contrast_estimate.overwrite = False
        subject_level.connect(model_estimate, 'spm_mat_file', contrast_estimate, 'spm_mat_file')
        subject_level.connect(model_estimate, 'beta_images', contrast_estimate, 'beta_images')
        subject_level.connect(
            model_estimate, 'residual_image', contrast_estimate, 'residual_image')

        smooth = Node(Smooth(), name = 'smooth')
        smooth.inputs.fwhm = self.fwhm
        smooth.overwrite = False
        subject_level.connect(contrast_estimate, 'con_images', smooth, 'in_files')

        # DataSink - store the wanted results in the wanted repository
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir
        subject_level.connect(
            smooth, 'smoothed_files', data_sink, f'{subject_level.name}.@con_images')
        subject_level.connect(
            contrast_estimate, 'spm_mat_file', data_sink, f'{subject_level.name}.@spm_mat_file')

        # Remove large files, if requested
        if Configuration()['pipelines']['remove_unused_data']:

            # Remove Node - Remove gunzip files once they are no longer needed
            remove_gunzip = MapNode(
                InterfaceFactory.create('remove_parent_directory'),
                name = 'remove_gunzip',
                iterfield = ['file_name']
                )

            # Add connections
            subject_level.connect([
                (data_sink, remove_gunzip, [('out_file', '_')]),
                (gunzip, remove_gunzip, [('out_file', 'file_name')])
                ])

        return subject_level

    def get_subject_level_outputs(self):
        """ Return the names of the files the subject level analysis is supposed to generate. """

        templates = [join(
            self.directories.output_dir,
            'subject_level', '_subject_id_{subject_id}', f'con_{contrast_id}.nii')\
            for contrast_id in self.contrast_list]
        templates += [join(
            self.directories.output_dir,
            'subject_level', '_subject_id_{subject_id}', 'SPM.mat')]

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

        return [
            self.get_group_level_analysis_single_group('equalRange'),
            self.get_group_level_analysis_single_group('equalIndifference'),
            self.get_group_level_analysis_group_comparison()
        ]

    def get_group_level_analysis_single_group(self, method):
        """
        Return a workflow for the group level analysis in the single group case.

        Parameters:
            - method: one of 'equalRange', 'equalIndifference'

        Returns:
            - group_level_analysis: nipype.WorkFlow
        """
        # Compute the number of participants used to do the analysis
        nb_subjects = len(self.subject_list)

        # Infosource - a function free node to iterate over the list of subject names
        info_source = Node(IdentityInterface(fields=['contrast_id']),
                          name = 'info_source')
        info_source.iterables = [('contrast_id', self.contrast_list)]

        # Select files from subject level analysis
        templates = {
            'contrasts': join(self.directories.output_dir,
                'subject_level', '_subject_id_*', 'scon_{contrast_id}.nii'),
            }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.sort_filelist = True
        select_files.inputs.base_directory = self.directories.dataset_dir

        # Datasink - save important files
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir

        # Function Node get_group_subjects
        #   Get subjects in the group and in the subject_list
        get_group_subjects = Node(Function(
            function = list_intersection,
            input_names = ['list_1', 'list_2'],
            output_names = ['out_list']
            ),
            name = 'get_group_subjects'
        )
        get_group_subjects.inputs.list_1 = get_group(method)
        get_group_subjects.inputs.list_2 = self.subject_list

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

        # One Sample T-Test Design - creates one sample T-Test Design
        onesamplettestdes = Node(OneSampleTTestDesign(), name = 'onesampttestdes')

        # EstimateModel - estimate the parameters of the model
        # Even for second level it should be 'Classical': 1.
        level2estimate = Node(EstimateModel(), name = 'level2estimate')
        level2estimate.inputs.estimation_method = {'Classical': 1}

        # EstimateContrast - estimates simple group contrast
        level2conestimate = Node(EstimateContrast(), name = 'level2conestimate')
        level2conestimate.inputs.group_contrast = True
        level2conestimate.inputs.contrasts = [
            ['Group', 'T', ['mean'], [1]], ['Group', 'T', ['mean'], [-1]]]

        # Threshold Node - Create thresholded maps
        threshold = MapNode(Threshold(), name = 'threshold',
            iterfield = ['stat_image', 'contrast_index'])
        threshold.inputs.use_fwe_correction = True
        threshold.inputs.height_threshold_type = 'p-value'
        threshold.inputs.force_activation = False
        threshold.inputs.height_threshold = 0.05
        threshold.inputs.extent_threshold = 5
        threshold.inputs.contrast_index = [1, 2]

        # Create the group level workflow
        group_level_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = f'group_level_analysis_{method}_nsub_{nb_subjects}')
        group_level_analysis.connect([
            (info_source, select_files, [('contrast_id', 'contrast_id')]),
            (select_files, get_contrasts, [('contrasts', 'input_str')]),
            (get_group_subjects, get_contrasts, [
                (('out_list', complete_subject_ids), 'elements')
                ]),
            (get_contrasts, onesamplettestdes, [
                (('out_list', clean_list), 'in_files')
                ]),
            #(select_files, onesamplettestdes, [('mask', 'explicit_mask_file')]),
            (onesamplettestdes, level2estimate, [('spm_mat_file', 'spm_mat_file')]),
            (level2estimate, level2conestimate, [
                ('spm_mat_file', 'spm_mat_file'),
                ('beta_images', 'beta_images'),
                ('residual_image', 'residual_image')
                ]),
            (level2conestimate, threshold, [
                ('spm_mat_file', 'spm_mat_file'),
                ('spmT_images', 'stat_image')
                ]),
            (level2estimate, data_sink, [
                ('mask_image', f'{group_level_analysis.name}.@mask')]),
            (level2conestimate, data_sink, [
                ('spm_mat_file', f'{group_level_analysis.name}.@spm_mat'),
                ('spmT_images', f'{group_level_analysis.name}.@T'),
                ('con_images', f'{group_level_analysis.name}.@con')]),
            (threshold, data_sink, [
                ('thresholded_map', f'{group_level_analysis.name}.@thresh')])
            ])

        return group_level_analysis

    def get_group_level_analysis_group_comparison(self):
        """
        Return a workflow for the group level analysis in the group comparison case.

        Returns:
            - group_level_analysis: nipype.WorkFlow
        """
        # Compute the number of participants used to do the analysis
        nb_subjects = len(self.subject_list)

        # Infosource - a function free node to iterate over the list of subject names
        info_source = Node(IdentityInterface(fields=['contrast_id']),
                          name = 'info_source')
        info_source.iterables = [('contrast_id', self.contrast_list)]

        # Select files from subject level analysis
        templates = {
            'contrasts': join(self.directories.output_dir,
                'subject_level', '_subject_id_*', 'scon_{contrast_id}.nii'),
            #'mask': join('derivatives/fmriprep/gr_mask_tmax.nii')
            }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.sort_filelist = True
        select_files.inputs.base_directory = self.directories.dataset_dir

        # Datasink - save important files
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir

        # Function Node get_group_subjects
        #   Get subjects in the group and in the subject_list
        get_equal_indifference_subjects = Node(Function(
            function = list_intersection,
            input_names = ['list_1', 'list_2'],
            output_names = ['out_list']
            ),
            name = 'get_equal_indifference_subjects'
        )
        get_equal_indifference_subjects.inputs.list_1 = get_group('equalIndifference')
        get_equal_indifference_subjects.inputs.list_2 = self.subject_list

        # Function Node get_group_subjects
        #   Get subjects in the group and in the subject_list
        get_equal_range_subjects = Node(Function(
            function = list_intersection,
            input_names = ['list_1', 'list_2'],
            output_names = ['out_list']
            ),
            name = 'get_equal_range_subjects'
        )
        get_equal_range_subjects.inputs.list_1 = get_group('equalRange')
        get_equal_range_subjects.inputs.list_2 = self.subject_list

        # Create a function to complete the subject ids out from the get_equal_*_subjects nodes
        #   If not complete, subject id '001' in search patterns
        #   would match all contrast files with 'con_0001.nii'.
        complete_subject_ids = lambda l : [f'_subject_id_{a}' for a in l]

        # Function Node elements_in_string
        #   Get contrast files for required subjects
        # Note : using a MapNode with elements_in_string requires using clean_list to remove
        #   None values from the out_list
        get_equal_indifference_contrasts = MapNode(Function(
            function = elements_in_string,
            input_names = ['input_str', 'elements'],
            output_names = ['out_list']
            ),
            name = 'get_equal_indifference_contrasts', iterfield = 'input_str'
        )
        get_equal_range_contrasts = MapNode(Function(
            function = elements_in_string,
            input_names = ['input_str', 'elements'],
            output_names = ['out_list']
            ),
            name = 'get_equal_range_contrasts', iterfield = 'input_str'
        )

        # Two Sample T-Test Design
        twosampttest = Node(TwoSampleTTestDesign(), name = 'twosampttest')

        # EstimateModel - estimate the parameters of the model
        # Even for second level it should be 'Classical': 1.
        level2estimate = Node(EstimateModel(), name = 'level2estimate')
        level2estimate.inputs.estimation_method = {'Classical': 1}

        # EstimateContrast - estimates simple group contrast
        level2conestimate = Node(EstimateContrast(), name = 'level2conestimate')
        level2conestimate.inputs.group_contrast = True
        level2conestimate.inputs.contrasts = [
            ['Eq range vs Eq indiff in loss', 'T', ['Group_{1}', 'Group_{2}'], [1, -1]]
        ]

        # Threshold Node - Create thresholded maps
        threshold = Node(Threshold(), name = 'threshold')
        threshold.inputs.use_fwe_correction = True
        threshold.inputs.height_threshold_type = 'p-value'
        threshold.inputs.force_activation = False
        threshold.inputs.height_threshold = 0.05
        threshold.inputs.contrast_index = 1
        threshold.inputs.extent_threshold = 5

        # Create the group level workflow
        group_level_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = f'group_level_analysis_groupComp_nsub_{nb_subjects}')
        group_level_analysis.connect([
            (info_source, select_files, [('contrast_id', 'contrast_id')]),
            (select_files, get_equal_range_contrasts, [('contrasts', 'input_str')]),
            (select_files, get_equal_indifference_contrasts, [('contrasts', 'input_str')]),
            (get_equal_range_subjects, get_equal_range_contrasts, [
                (('out_list', complete_subject_ids), 'elements')
                ]),
            (get_equal_indifference_subjects, get_equal_indifference_contrasts, [
                (('out_list', complete_subject_ids), 'elements')
                ]),
            (get_equal_range_contrasts, twosampttest, [
                (('out_list', clean_list), 'group1_files')
                ]),
            (get_equal_indifference_contrasts, twosampttest, [
                (('out_list', clean_list), 'group2_files')
                ]),
            #(select_files, twosampttest, [('mask', 'explicit_mask_file')]),
            (twosampttest, level2estimate, [('spm_mat_file', 'spm_mat_file')]),
            (level2estimate, level2conestimate, [
                ('spm_mat_file', 'spm_mat_file'),
                ('beta_images', 'beta_images'),
                ('residual_image', 'residual_image')
                ]),
            (level2conestimate, threshold, [
                ('spm_mat_file', 'spm_mat_file'),
                ('spmT_images', 'stat_image')
                ]),
            (level2estimate, data_sink, [
                ('mask_image', f'{group_level_analysis.name}.@mask')]),
            (level2conestimate, data_sink, [
                ('spm_mat_file', f'{group_level_analysis.name}.@spm_mat'),
                ('spmT_images', f'{group_level_analysis.name}.@T'),
                ('con_images', f'{group_level_analysis.name}.@con')]),
            (threshold, data_sink, [
                ('thresholded_map', f'{group_level_analysis.name}.@thresh')])
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
                'con_0001.nii', 'mask.nii', 'SPM.mat', 'spmT_0001.nii', 'spmT_0001_thr.nii'
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
                '_contrast_id_0002', '_threshold1', 'spmT_0002_thr.nii'),
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
                '_contrast_id_0002', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0001.nii'),
            # Hypothesis 9
            join(f'group_level_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0001.nii')
        ]
        return [join(self.directories.output_dir, f) for f in files]
