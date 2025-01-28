#!/usr/bin/python
# coding: utf-8

""" Write the work of NARPS team E3B6 using Nipype """
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

class PipelineTeamE3B6(Pipeline):
    """ A class that defines the pipeline of team E3B6 """

    def __init__(self):
        super().__init__()
        self._fwhm = 6.0
        self.team_id = 'E3B6'

        # Define contrasts
        self.contrast_list = ['0001', '0002', '0003']
        conditions = ['trialxparam_gain^1', 'trialxparam_loss^1']
        self.subject_level_contrasts = [
            ('positive_effect_of_gain', 'T', conditions, [0, 1, 0]),
            ('negative_effect_of_loss', 'T', conditions, [0, 0, -1]),
            ('positive_effect_of_loss', 'T', conditions, [0, 0, 1])
        ]

    def get_preprocessing(self):
        """
        Create the preprocessing workflow.
            Preprocessing order : fMRIprep + smoothing
            Smoothing :  SPM 12 ; v7487 fixed Gaussian kernel in MNI volume FWHM = 6 mm
        """
        # Create preprocessing workflow
        preprocessing =  Workflow(base_dir = self.directories.working_dir, name = 'preprocessing')

        # IDENTITY INTERFACE - To iterate on subjects
        info_source = Node(IdentityInterface(fields=['subject_id', 'run_id']), name='info_source')
        info_source.iterables = [
            ('subject_id', self.subject_list),
            ('run_id', self.run_list),
            ]

        # SELECT FILES - Select necessary files
        file_templates = {
            'func': join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_}_bold_space-MNI152NLin2009cAsym_preproc'
            )
        }
        select_files = Node(SelectFiles(file_templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir
        preprocessing.connect(info_source, 'subject_id', select_files, 'subject_id')
        preprocessing.connect(info_source, 'run_id', select_files, 'run_id')

        # GUNZIP - gunzip files because SPM do not use .nii.gz files
        gunzip_func = MapNode(Gunzip(), name='gunzip_func', iterfield=['in_file'])
        preprocessing.connect(select_files, 'func', gunzip_func, 'in_file')

        # SMOOTH - smoothing node
        # https://github.com/Remi-Gau/NARPS_CPPL/blob/v0.0.2/subfun/smooth_batch.m
        smoothing = Node(Smooth(), name = 'smooth')
        smoothing.inputs.fwhm = self.fwhm
        smoothing.inputs.data_type = 0
        preprocessing.connect(gunzip_func, 'out_file', smoothing, 'in_files')

        # DataSink Node - store the wanted results in the wanted repository
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir
        preprocessing.connect(
            smoothing, 'smoothed_files', data_sink, 'preprocessing.@output_image')

        # Remove large files, if requested
        if Configuration()['pipelines']['remove_unused_data']:
            # Merge Node - Merge func file names to be removed after datasink node is performed
            merge_removable_func_files = Node(Merge(2), name = 'merge_removable_func_files')
            merge_removable_func_files.inputs.ravel_inputs = True

            # Function Nodes remove_files - Remove sizeable func files once they aren't needed
            remove_func_after_datasink = MapNode(Function(
                function = remove_parent_directory,
                input_names = ['_', 'file_name'],
                output_names = []
                ), name = 'remove_func_after_datasink', iterfield = 'file_name')
            preprocessing.connect([
                (gunzip_func, merge_removable_func_files, [('out_file', 'in1')]),
                (smoothing, merge_removable_func_files, [('smoothed_files', 'in2')]),
                (merge_removable_func_files, remove_func_after_datasink, [('out', 'file_name')]),
                (data_sink, remove_func_after_datasink, [('out_file', '_')])
            ])

        return preprocessing

    def get_preprocessing_outputs(self):
        """ Return the names of the files the preprocessing is supposed to generate. """

        # Outputs from preprocessing (smoothed func files)
        parameters = {
            'subject_id': self.subject_list,
            'run_id': self.run_list,
        }
        parameter_sets = product(*parameters.values())
        template = join(self.directories.output_dir, 'preprocessing',
            '_run_id_{run_id}_subject_id_{subject_id}',
            'ssub-{subject_id}_task-MGT_run-{run_id}_bold.nii')
        return_list += [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

        return return_list

    def get_run_level_analysis(self):
        """ No run level analysis was done by team E3B6 (as for all the pipelines using SPM) """
        return None

    def get_nuisance_regressors():
        """
        Nuisance regressors: as specified above included white matter signal, CSF signal, framewise displacement, and censoring regressors
        """
        return

    def get_subject_infos(event_file: str, short_run_id: str):
        """
        We used an approach similar to Tom et al 2007.
        All trials were modeled using a single condition (with a duration of 4 secs)
        and three additional parametric regressors were included :
            a) param_loss: a regressor modulated by the loss vallue associated to that trial
            b) param_gain: regressor modulated by the gain vallue associated to that trial
            c) the Euclidean distance of the gain/loss combination from the indifference line
                (i.e. assuming lambda=2 and a linear value function).

        All loss/gain/EV values were mean-centered in a runwise fashion before being used
        to modulate the trial regressor.

        Button presses were modelled as a separate condition of no interest:
        they were modelled as delta functions (duration = 0) with onsets defined by the trial onset + reaction time relative to that trial.
        They were then convolved with the HRF (see below).

        Trials with no reponse or very short reponses (< 500 ms) were modelled separately in another condition of no interest.

        The HRF was modelled using SPM's canonical HRF function.

        Drift regressors were the DCT basis included in the GLM (SPM default 128 secs high-pass filter)

        Orthogonalization: the SPM automatic orthogonalization of parametric regressors was disabled.
        In other words the gain, loss and expected value modualted regressors were not orthogonalized with respect to the unmodulated regressors
        (as we were not interested in the activations due to this regressors per se).

        Create Bunch for specifySPMModel.

        Parameters :
        - event_file: str, event files for the run
        - short_run_id: str, shorten version of the run id

        Returns :
        Bunch for 1st level analysis
        """
        from numpy import mean
        from nipype.interfaces.base import Bunch

        # Parameters and regressors for the main trial condition
        onsets = []
        durations = []
        param_loss = []
        param_gain = []
        expected_value = []

        # Parameters for the button presses condition
        onsets_button = []
        durations_button = []

        # Parameters for the no response condition
        onsets_no_response = []
        durations_no_response = []

        # Read events file
        with open(event_file, 'rt') as file:
            next(file)  # skip the header

            for line in file:
                info = line.strip().split()

                if float(info[4]) < 0.5 or 'NoResp' in info[5]:
                    # No response condition
                    onsets_no_response.append(float(info[0]) + float(info[4]))
                    durations_no_response.append(0.0) # TODO : duration not specified
                else:
                    # Button presses condition
                    onsets_button.append(float(info[0]) + float(info[4]))
                    durations_button.append(0.0)

                    # Main trial
                    onsets.append(float(info[0]))
                    durations.append(4.0)
                    param_gain.append(float(info[2]))
                    param_loss.append(float(info[3]))
                    # Compute expected value
                    # see https://github.com/Remi-Gau/NARPS_CPPL/blob/1835647390f94625b5de2470b7a975b8e2454a92/step_3_run_first_level.m#L107
                    expected_value.append(abs(0.5 * float(info[2]) - float(info[3])) / 1.25**.5)

        # Mean center regressors
        param_gain = list(param_gain - mean(param_gain))
        param_loss = list(param_loss - mean(param_loss))
        expected_value = list(expected_value - mean(expected_value))

        # Return data formatted as a Bunch
        return Bunch(
            conditions = ['trial', 'button_press', 'no_response'],
            onsets = [onsets, onsets_button, onsets_no_response],
            durations = [durations, durations_button, durations_no_response],
            amplitudes = None,
            tmod = None,
            pmod = [
                Bunch(
                    name = ['param_loss', 'param_gain', 'expected_value'],
                    poly = [1, 1, 1],
                    param = [param_loss, param_gain, expected_value],
                ),
                None,
                None
            ],
            regressor_names = None,
            regressors = None
            )

    def get_subject_level_analysis(self):
        """ Return a Nipype workflow describing the subject level analysis part of the pipeline
            https://github.com/Remi-Gau/NARPS_CPPL/blob/v0.0.2/step_3_run_first_level.m
        """

        # Init workflow
        subject_level_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = 'subject_level_analysis'
            )

        # IDENTITY INTERFACE - To iterate on subjects
        info_source = Node(IdentityInterface(fields = ['subject_id']), name = 'info_source')
        info_source.iterables = [('subject_id', self.subject_list)]

        # SELECT FILES - to select necessary files
        templates = {
            'func': join('preprocessing', '_run_id_*_subject_id_{subject_id}',
                'ssub-{subject_id}_task-MGT_run-*_bold.nii',
            ),
            'mask': join(self.directories.dataset_dir, 'derivatives', 'fmriprep',
                'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-*_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz'),
            'event': join(
                self.directories.dataset_dir, 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-*_events.tsv',
            )
        }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.results_dir
        subject_level_analysis.connect(info_source, 'subject_id', select_files, 'subject_id')

        # Compute explicit mask
        """
        The spatial region modeled included the voxels of the brain mask from fMRIprep: it was used as an explicit mask in the subject-level GLM in SPM. We computed the union of the brainmasks computed for each functional run by fMRIprep.
        The SPM threshold to define the implicit mask was set to 0 (instead of the 0.8 default).
        https://github.com/Remi-Gau/NARPS_CPPL/blob/v0.0.2/subfun/create_mask.m
        """

        # FUNCTION node get_subject_information - get subject specific condition information
        subject_infos = Node(Function(
                function = self.get_subject_infos,
                input_names = ['event_files', 'runs'],
                output_names = ['subject_info']
            ),
            name = 'subject_infos',
        )
        subject_infos.inputs.runs = self.run_list
        subject_level_analysis.connect(select_files, 'events', subject_information, 'event_file')

        # SPECIFY MODEL - Generates SPM-specific Model
        specify_model = Node(SpecifySPMModel(), name = 'specify_model')
        specify_model.inputs.concatenate_runs = False
        specify_model.inputs.input_units = 'secs'
        specify_model.inputs.output_units = 'secs'
        specify_model.inputs.time_repetition = TaskInformation()['RepetitionTime']
        specify_model.inputs.high_pass_filter_cutoff = 128
        subject_level_analysis.connect(select_files, 'func', specify_model, 'functional_runs')
        subject_level_analysis.connect(
            subject_information, 'subject_info', specify_model, 'subject_info')
        subject_level_analysis.connect( , , specify_model, 'realignment_parameters')

        # LEVEL1DESIGN - Generates an SPM design matrix
        model_design = Node(Level1Design(), name = 'model_design')
        model_design.inputs.bases = {'hrf': {'derivs': [1, 1]}}
        model_design.inputs.timing_units = 'secs'
        model_design.inputs.interscan_interval = TaskInformation()['RepetitionTime']
        model_design.inputs.volterra_expansion_order = 2
        subject_level_analysis.connect(specify_model, 'session_info', model_design, 'session_info')

        # ESTIMATE MODEL - estimate the parameters of the model
        model_estimate = Node(EstimateModel(), name = 'model_estimate')
        model_estimate.inputs.estimation_method = {'Classical': 1}
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
            model_estimate, 'residual_image', contrast_estimate,'residual_image')

        # DATASINK - store the wanted results in the wanted repository
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir
        subject_level_analysis.connect(
            contrast_estimate, 'con_images', data_sink, 'subject_level_analysis.@con_images')
        subject_level_analysis.connect(
            contrast_estimate, 'spmT_images', data_sink, 'subject_level_analysis.@spmT_images')
        subject_level_analysis.connect(
            contrast_estimate, 'spm_mat_file', data_sink, 'subject_level_analysis.@spm_mat_file')

        return subject_level_analysis

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
        https://github.com/Remi-Gau/NARPS_CPPL/blob/v0.0.2/step_4_run_second_level.m

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

        # Clusternig used https://github.com/spisakt/pTFCE/releases/tag/v0.1.3





        # [INFO] The following part stays the same for all preprocessing pipelines

        # Infosource node - iterate over the list of contrasts generated
        # by the subject level analysis
        info_source = Node(
            IdentityInterface(
                fields=['contrast_id', 'subjects'],
                subjects=self.subject_list
            ),
            name='info_source',
        )
        info_source.iterables = [('contrast_id', self.contrast_list)]

        # Templates to select files node
        # [TODO] Change the name of the files depending on the filenames
        # of results of first level analysis
        templates = {
            'contrast': join(
                self.directories.results_dir,
                'subject_level_analysis',
                '_subject_id_*',
                'complete_filename_{contrast_id}_complete_filename.nii',
            ),
            'participants': join(
                self.directories.dataset_dir,
                'participants.tsv'
            )
        }
        select_files = Node(
            SelectFiles(
                templates,
                base_directory=self.directories.results_dir,
                force_list=True
            ),
            name='select_files',
        )

        # Datasink node - to save important files
        data_sink = Node(
            DataSink(base_directory=self.directories.output_dir),
            name='data_sink',
        )

        # Contrasts node - to select subset of contrasts
        sub_contrasts = Node(
            Function(
                input_names=['file_list', 'method', 'subject_list', 'participants_file'],
                output_names=[
                    'equalIndifference_id',
                    'equalRange_id',
                    'equalIndifference_files',
                    'equalRange_files',
                ],
                function=self.get_subset_contrasts,
            ),
            name='sub_contrasts',
        )
        sub_contrasts.inputs.method = method

        # [INFO] The following part has to be modified with nodes of the pipeline

        # [TODO] For each node, replace 'node_name' by an explicit name, and use it for both:
        #   - the name of the variable in which you store the Node object
        #   - the 'name' attribute of the Node
        # [TODO] The node_function refers to a NiPype interface that you must import
        # at the beginning of the file.
        node_name = Node(
            node_function,
            name='node_name'
        )

        # [INFO] The following part defines the nipype workflow and the connections between nodes

        # Compute the number of participants used to do the analysis
        nb_subjects = len(self.subject_list)

        # Declare the workflow
        group_level_analysis = Workflow(
            base_dir=self.directories.working_dir,
            name=f'group_level_analysis_{method}_nsub_{nb_subjects}'
        )
        group_level_analysis.connect(
            [
                (
                    info_source,
                    select_files,
                    [('contrast_id', 'contrast_id')],
                ),
                (info_source, sub_contrasts, [('subjects', 'subject_list')]),
                (
                    select_files,
                    sub_contrasts,
                    [('contrast', 'file_list'), ('participants', 'participants_file')],
                ),  # Complete with other links between nodes
            ]
        )

        # [INFO] Here we define the contrasts used for the group level analysis, depending on the
        # method used.
        if method in ('equalRange', 'equalIndifference'):
            contrasts = [('Group', 'T', ['mean'], [1]), ('Group', 'T', ['mean'], [-1])]

        elif method == 'groupComp':
            contrasts = [
                ('Eq range vs Eq indiff in loss', 'T', ['Group_{1}', 'Group_{2}'], [1, -1])
            ]

        # [INFO] Here we simply return the created workflow
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
                '_contrast_id_0002', '_threshold0', 'spmT_0002_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0002.nii'),
            # Hypothesis 6
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0002', '_threshold1', 'spmT_0002_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0002', 'spmT_0002.nii'),
            # Hypothesis 7
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0003', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_0003', 'spmT_0001.nii'),
            # Hypothesis 8
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0003', '_threshold1', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_0003', 'spmT_0001.nii'),
            # Hypothesis 9
            join(f'group_level_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_0003', '_threshold0', 'spmT_0001_thr.nii'),
            join(f'group_level_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_0003', 'spmT_0001.nii')
        ]
        return [join(self.directories.output_dir, f) for f in files]
