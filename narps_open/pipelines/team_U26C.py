#!/usr/bin/python
# coding: utf-8

""" Write the work of NARPS' team U26C using Nipype """

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
from narps_open.utils.configuration import Configuration

class PipelineTeamU26C(Pipeline):
    """ A class that defines the pipeline of team U26C. """

    def __init__(self):
        super().__init__()
        self.fwhm = 5.0
        self.team_id = 'U26C'

        gamble = [f'gamble_run{r}' for r in range(1, len(self.run_list) + 1)]
        gain = [f'gamble_run{r}xgain_run{r}^1' for r in range(1, len(self.run_list) + 1)]
        loss = [f'gamble_run{r}xloss_run{r}^1' for r in range(1, len(self.run_list) + 1)]

        self.subject_level_contrasts = [
            ['gamble', 'T', gamble, [1, 1, 1, 1]],
            ['gain', 'T', gain, [1, 1, 1, 1]],
            ['loss', 'T', loss, [1, 1, 1, 1]]
        ]

    def get_preprocessing(self):
        """ No preprocessing has been done by team U26C """
        return None

    def get_run_level_analysis(self):
        """ No run level analysis has been done by team U26C """
        return None

    # @staticmethod # Starting python 3.10, staticmethod should be used here
    # Otherwise it produces a TypeError: 'staticmethod' object is not callable
    def get_subject_information(event_files: list):
        """ Create Bunchs for SpecifySPMModel.

        Parameters :
        - event_files: list of str, list of events files (one per run) for the subject

        Returns :
        - subject_information : list of Bunch for 1st level analysis.
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
                    conditions = [trial_key],
                    onsets = [onsets[trial_key]],
                    durations = [durations[trial_key]],
                    amplitudes = None,
                    tmod = None,
                    pmod = [Bunch(
                        name = [gain_key, loss_key],
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
        from numpy import array, transpose, insert, diff

        # Open original confounds file
        data_frame = read_csv(filepath, sep = '\t', header=0)

        # Extract confounds we want to use for the model
        retained_parameters = DataFrame(transpose(array([
            data_frame['CSF'], data_frame['WhiteMatter'],
            data_frame['X'], data_frame['Y'], data_frame['Z'],
            data_frame['RotX'], data_frame['RotY'], data_frame['RotZ'],
            insert(diff(data_frame['X']), 0, 0),
            insert(diff(data_frame['Y']), 0, 0),
            insert(diff(data_frame['Z']), 0, 0),
            insert(diff(data_frame['RotX']), 0, 0),
            insert(diff(data_frame['RotY']), 0, 0),
            insert(diff(data_frame['RotZ']), 0, 0)
            ]
        )))

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
            - subject_level_analysis : nipype.WorkFlow
        """
        # Identity interface Node - to iterate over subject_id and run
        infosource = Node(interface=IdentityInterface(fields=['subject_id']),
            name = 'infosource')
        infosource.iterables = [('subject_id', self.subject_list)]

        # Select files from derivatives
        templates = {
            'func': join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-*_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'),
            'confounds' : join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-*_bold_confounds.tsv'),
            'events': join('sub-{subject_id}', 'func',
               'sub-{subject_id}_task-MGT_run-*_events.tsv')
        }        
        selectderivs = Node(SelectFiles(templates), name = 'selectderivs')
        selectderivs.inputs.base_directory = self.directories.dataset_dir
        selectderivs.inputs.sort_filelist = True

        # Gunzip - gunzip files because SPM do not use .nii.gz files
        gunzip = MapNode(Gunzip(), name='gunzip', iterfield=['in_file'])

        # Smooth warped functionals.
        smooth = Node(Smooth(), name = 'smooth')
        smooth.inputs.fwhm = self.fwhm
        smooth.overwrite = False

        # Function node get_subject_information - get subject specific condition information
        getsubinforuns = Node(Function(
            function = self.get_subject_information,
            input_names = ['event_files'],
            output_names = ['subject_info']
            ),
            name = 'getsubinforuns')

        # Function node get_confounds_file - get confounds files
        confounds = MapNode(Function(
            function = self.get_confounds_file,
            input_names = ['filepath', 'subject_id', 'run_id', 'working_dir'],
            output_names = ['confounds_file']),
            name = 'confounds', iterfield = ['filepath', 'run_id'])
        confounds.inputs.working_dir = self.directories.working_dir
        confounds.inputs.run_id = self.run_list

        modelspec = Node(SpecifySPMModel(), name = 'modelspec')
        modelspec.inputs.concatenate_runs = False
        modelspec.inputs.input_units = 'secs'
        modelspec.inputs.output_units = 'secs'
        modelspec.inputs.time_repetition = TaskInformation()['RepetitionTime']
        modelspec.inputs.high_pass_filter_cutoff = 128
        modelspec.overwrite = False

        level1design = Node(Level1Design(), name = 'level1design')
        level1design.inputs.bases = {'hrf': {'derivs': [0, 0]}}
        level1design.inputs.timing_units = 'secs'
        level1design.inputs.interscan_interval = TaskInformation()['RepetitionTime']
        level1design.overwrite = False

        level1estimate = Node(EstimateModel(), name = 'level1estimate')
        level1estimate.inputs.estimation_method = {'Classical': 1}
        level1estimate.overwrite = False

        contrast_estimate = Node(EstimateContrast(), name = 'contraste_estimate')
        contrast_estimate.inputs.contrasts = self.subject_level_contrasts
        contrast_estimate.config = {'execution': {'remove_unnecessary_outputs': False}}
        contrast_estimate.overwrite = False

        subject_level_analysis = Workflow(
            base_dir = self.directories.working_dir, name = 'subject_level_analysis'
            )
        subject_level_analysis.connect([
            (infosource, selectderivs, [('subject_id', 'subject_id')]),
            (infosource, confounds, [('subject_id', 'subject_id')]),
            (selectderivs, getsubinforuns, [('events', 'event_files')]),
            (selectderivs, gunzip, [('func', 'in_file')]),
            (selectderivs, confounds, [('confounds', 'filepath')]),
            (gunzip, smooth, [('out_file', 'in_files')]),
            (getsubinforuns, modelspec, [('subject_info', 'subject_info')]),
            (confounds, modelspec, [('confounds_file', 'realignment_parameters')]),
            (smooth, modelspec, [('smoothed_files', 'functional_runs')]),
            (modelspec, level1design, [('session_info', 'session_info')]),
            (level1design, level1estimate, [('spm_mat_file', 'spm_mat_file')]),
            (level1estimate, contrast_estimate,[
                ('spm_mat_file', 'spm_mat_file'),
                ('beta_images', 'beta_images'),
                ('residual_image', 'residual_image')])
            ])

        # Remove large files, if requested
        if Configuration()['pipelines']['remove_unused_data']:

            # Remove Node - Remove gunzip files once they are no longer needed
            remove_gunzip = MapNode(
                InterfaceFactory.create('remove_parent_directory'),
                name = 'remove_gunzip',
                iterfield = ['file_name']
                )

            # Remove Node - Remove smoothed files once they are no longer needed
            remove_smooth = MapNode(
                InterfaceFactory.create('remove_parent_directory'),
                name = 'remove_smooth',
                iterfield = ['file_name']
                )

            # Add connections
            subject_level_analysis.connect([
                (smooth, remove_gunzip, [('smoothed_files', '_')]),
                (gunzip, remove_gunzip, [('out_file', 'file_name')]),
                (modelspec, remove_smooth, [('session_info', '_')]),
                (smooth, remove_smooth, [('smoothed_files', 'file_name')])
                ])

        return subject_level_analysis

    def get_subject_level_outputs(self):
        """ Return the names of the files the subject level analysis is supposed to generate. """

        # Handle gain files
        templates = [join(
            self.directories.output_dir,
            'subject_level_analysis_gain', '_subject_id_{subject_id}', 'con_0001.nii')]
        templates += [join(
            self.directories.output_dir,
            'subject_level_analysis_gain', '_subject_id_{subject_id}', 'SPM.mat')]
        templates += [join(
            self.directories.output_dir,
            'subject_level_analysis_gain', '_subject_id_{subject_id}', 'spmT_0001.nii')]

        # Handle loss files
        contrast_list = ['0001', '0002']
        templates += [join(
            self.directories.output_dir,
            'subject_level_analysis_loss', '_subject_id_{subject_id}', f'con_{contrast_id}.nii')\
            for contrast_id in contrast_list]
        templates += [join(
            self.directories.output_dir,
            'subject_level_analysis_loss', '_subject_id_{subject_id}', 'SPM.mat')]
        templates += [join(
            self.directories.output_dir,
            'subject_level_analysis_loss', '_subject_id_{subject_id}', f'spmT_{contrast_id}.nii')\
            for contrast_id in contrast_list]

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

    def get_group_level_analysis_sub_workflow(self, method):
        """
        Return a workflow for the group level analysis.

        Parameters:
            - method: one of 'equalRange', 'equalIndifference' or 'groupComp'

        Returns:
            - group_level_analysis: nipype.WorkFlow
        """

    def get_group_level_outputs(self):
        """ Return all names for the files the group level analysis is supposed to generate. """

        # Handle equalRange and equalIndifference

        ## Contrast id 0001
        parameters = {
            'method': ['equalRange', 'equalIndifference'],
            'file': [
                'con_0001.nii', 'con_0002.nii', 'mask.nii', 'SPM.mat',
                'spmT_0001.nii', 'spmT_0002.nii',
                join('_threshold0', 'spmT_0001_thr.nii'), join('_threshold1', 'spmT_0002_thr.nii')
                ],
            'model_type' : ['gain', 'loss'],
            'nb_subjects' : [str(len(self.subject_list))]
        }

        parameter_sets = product(*parameters.values())
        template = join(
            self.directories.output_dir,
            'group_level_analysis_{method}_nsub_{nb_subjects}',
            '_contrast_id_0001_model_type_{model_type}',
            '{file}'
            )

        return_list = [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

        ## Contrast id 0002
        parameters = {
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
            '_contrast_id_0002_model_type_loss',
            '{file}'
            )

        return_list += [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

        # Handle groupComp
        parameters = {
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
            '_contrast_id_0001_model_type_loss',
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
                '_contrast_id_0001_model_type_gain', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_gain', 'spmT_0001.nii'),
            # Hypothesis 2
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_gain', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_gain', 'spmT_0001.nii'),
            # Hypothesis 3
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_gain', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_gain', 'spmT_0001.nii'),
            # Hypothesis 4
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_gain', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_gain', 'spmT_0001.nii'),
            # Hypothesis 5
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_loss', '_threshold1', 'spmT_0002_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_loss', 'spmT_0002.nii'),
            # Hypothesis 6
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_loss', '_threshold1', 'spmT_0002_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_loss', 'spmT_0002.nii'),
            # Hypothesis 7
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_loss', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_loss', 'spmT_0001.nii'),
            # Hypothesis 8
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_loss', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_loss', 'spmT_0001.nii'),
            # Hypothesis 9
            join(f'group_level_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_loss', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_0001_model_type_loss', 'spmT_0001.nii')
        ]
        return [join(self.directories.output_dir, f) for f in files]
