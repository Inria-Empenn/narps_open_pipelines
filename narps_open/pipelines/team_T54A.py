#!/usr/bin/python
# coding: utf-8

""" Write the work of NARPS team T54A using Nipype """

from os.path import join
from itertools import product

from nipype import Workflow, Node
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.fsl import (
    BET, IsotropicSmooth, Level1Design, FEATModel, L2Model, Merge, FLAMEO,
    FILMGLS, Randomise, MultipleRegressDesign, FSLCommand
    )
from nipype.algorithms.modelgen import SpecifyModel

from narps_open.pipelines import Pipeline
from narps_open.data.task import TaskInformation

# Setup FSL
FSLCommand.set_default_output_type('NIFTI_GZ')

class PipelineTeamT54A(Pipeline):
    """ A class that defines the pipeline of team T54A. """

    def __init__(self):
        super().__init__()
        self.fwhm = 4.0
        self.team_id = 'T54A'
        self.contrast_list = ['1', '2']

    def get_preprocessing(self):
        """ No preprocessing has been done by team T54A """
        return None

    def get_session_infos(event_file):
        """
        Create Bunchs for specifyModel.

        Parameters :
        - event_file : str, file corresponding to the run and the subject to analyze

        Returns :
        - subject_info : list of Bunch for 1st level analysis.
        """
        from nipype.interfaces.base import Bunch

        condition_names = ['trial', 'gain', 'loss', 'difficulty', 'response']
        onset = {}
        duration = {}
        amplitude = {}

        for condition in condition_names:
            # Create dictionary items with empty lists
            onset.update({condition : []})
            duration.update({condition : []})
            amplitude.update({condition : []})

        with open(event_file, 'rt') as file:
            next(file)  # skip the header

            for line in file:
                info = line.strip().split()
                # Creates list with onsets, duration and loss/gain
                # for amplitude (FSL)
                for condition in condition_names:
                    if info[5] != 'NoResp':
                        if condition == 'gain':
                            onset[condition].append(float(info[0]))
                            duration[condition].append(float(info[4]))
                            amplitude[condition].append(float(info[2]))
                        elif condition == 'loss':
                            onset[condition].append(float(info[0]))
                            duration[condition].append(float(info[4]))
                            amplitude[condition].append(float(info[3]))
                        elif condition == 'trial':
                            onset[condition].append(float(info[0]))
                            duration[condition].append(float(info[4]))
                            amplitude[condition].append(float(1))
                        elif condition == 'difficulty':
                            onset[condition].append(float(info[0]))
                            duration[condition].append(float(info[4]))
                            amplitude[condition].append(
                                abs(0.5 * float(info[2]) - float(info[3]))
                                )
                        elif condition == 'response':
                            onset[condition].append(float(info[0]) + float(info[4]))
                            duration[condition].append(float(0))
                            amplitude[condition].append(float(1))
                    else:
                        if condition=='missed':
                            onset[condition].append(float(info[0]))
                            duration[condition].append(float(0))

        return [
            Bunch(
                conditions = condition_names,
                onsets = [onset[k] for k in condition_names],
                durations = [duration[k] for k in condition_names],
                amplitudes = [amplitude[k] for k in condition_names],
                regressor_names = None,
                regressors = None)
            ]

    def get_parameters_file(filepath, subject_id, run_id, working_dir):
        """
        Create a tsv file with only desired parameters per subject per run.

        Parameters :
        - filepath : path to subject parameters file (i.e. one per run)
        - subject_id : subject for whom the 1st level analysis is made
        - run_id: run for which the 1st level analysis is made
        - working_dir: str, name of the directory for intermediate results

        Return :
        - parameters_file : paths to new files containing only desired parameters.
        """
        from os import makedirs
        from os.path import join, isdir

        from pandas import read_csv, DataFrame
        from numpy import array, transpose

        data_frame = read_csv(filepath, sep = '\t', header=0)
        if 'NonSteadyStateOutlier00' in data_frame.columns:
            temp_list = array([
                data_frame['X'], data_frame['Y'], data_frame['Z'],
                data_frame['RotX'], data_frame['RotY'], data_frame['RotZ'],
                data_frame['NonSteadyStateOutlier00']])
        else:
            temp_list = array([
                data_frame['X'], data_frame['Y'], data_frame['Z'],
                data_frame['RotX'], data_frame['RotY'], data_frame['RotZ']])
        retained_parameters = DataFrame(transpose(temp_list))

        parameters_file = join(working_dir, 'parameters_file',
            f'parameters_file_sub-{subject_id}_run{run_id}.tsv')

        makedirs(join(working_dir, 'parameters_file'), exist_ok = True)

        with open(parameters_file, 'w') as writer:
            writer.write(retained_parameters.to_csv(
                sep = '\t', index = False, header = False, na_rep = '0.0'))

        return parameters_file

    def get_contrasts():
        """
        Create a list of tuples that represent contrasts.
        Each contrast is in the form :
        (Name,Stat,[list of condition names],[weights on those conditions])

        Returns:
            - contrasts: list of tuples, list of contrasts to analyze
        """
        # List of condition names
        conditions = ['trial', 'gain', 'loss']

        # Create contrasts
        gain = ('gain', 'T', conditions, [0, 1, 0])
        loss = ('loss', 'T', conditions, [0, 0, 1])

        # Contrast list
        return [gain, loss]

    def remove_smoothed_files(_, subject_id, run_id, working_dir):
        """
        This method is used in a Function node to fully remove
        the files generated by the smoothing node, once they aren't needed anymore.

        Parameters:
        - _: Node input only used for triggering the Node
        - subject_id: str, id of the subject from which to remove the file
        - run_id: str, id of the run from which to remove the file
        - working_dir: str, path to the working dir
        """
        from shutil import rmtree
        from os.path import join

        try:
            rmtree(join(
                working_dir, 'l1_analysis',
                f'_run_id_{run_id}_subject_id_{subject_id}', 'smooth')
            )
        except OSError as error:
            print(error)
        else:
            print('The directory is deleted successfully')

    def get_run_level_analysis(self):
        """
        Create the run level analysis workflow.

        Returns:
            - run_level_analysis : nipype.WorkFlow
        """
        # IdentityInterface Node - To iterate on subject and runs
        information_source = Node(IdentityInterface(
            fields = ['subject_id', 'run_id']),
            name = 'information_source')
        information_source.iterables = [
            ('subject_id', self.subject_list),
            ('run_id', self.run_list)
            ]

        # SelectFiles - to select necessary files
        template = {
            # Parameter file
            'param' : join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_confounds.tsv'),
            # Functional MRI
            'func' : join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
            'sub-{subject_id}_task-MGT_run-{run_id}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'
            ),
            # Event file
            'event' : join('sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_events.tsv')
        }
        select_files = Node(SelectFiles(template), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir

        # DataSink Node - store the wanted results in the wanted directory
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir

        # BET Node - Skullstripping data
        skull_stripping_func = Node(BET(), name = 'skull_stripping_func')
        skull_stripping_func.inputs.frac = 0.1
        skull_stripping_func.inputs.functional = True
        skull_stripping_func.inputs.mask = True

        # IsotropicSmooth Node - Smoothing data
        smoothing_func = Node(IsotropicSmooth(), name = 'smoothing_func')
        smoothing_func.inputs.fwhm = self.fwhm # TODO : Previously set to 6 mm ?

        # Function Node get_subject_infos - Get subject specific condition information
        subject_information = Node(Function(
            function = self.get_session_infos,
            input_names = ['event_file'],
            output_names = ['subject_info']
            ), name = 'subject_information')

        # SpecifyModel Node - Generate run level model
        specify_model = Node(SpecifyModel(), name = 'specify_model')
        specify_model.inputs.high_pass_filter_cutoff = 100
        specify_model.inputs.input_units = 'secs'
        specify_model.inputs.time_repetition = TaskInformation()['RepetitionTime']

        # Funcion Node get_contrasts - Get the list of contrasts
        contrasts = Node(Function(
            function = self.get_contrasts,
            input_names = [],
            output_names = ['contrasts']
            ), name = 'contrasts')

        # Function Node get_parameters_file - Get files with movement parameters
        parameters = Node(Function(
            function = self.get_parameters_file,
            input_names = ['filepath', 'subject_id', 'run_id', 'working_dir'],
            output_names = ['parameters_file']),
            name = 'parameters')
        parameters.inputs.working_dir = self.directories.working_dir

        # Level1Design Node - Generate files for run level computation
        model_design = Node(Level1Design(), name = 'model_design')
        model_design.inputs.bases = {'dgamma':{'derivs' : True}}
        model_design.inputs.interscan_interval = TaskInformation()['RepetitionTime']
        model_design.inputs.model_serial_correlations = True

        # FEATModel Node - Generate run level model
        model_generation = Node(FEATModel(), name = 'model_generation')

        # FILMGLS Node - Estimate first level model
        model_estimate = Node(FILMGLS(), name = 'model_estimate')

        # Function node remove_smoothed_files - remove output of the smooth node
        remove_smoothed_files = Node(Function(
            function = self.remove_smoothed_files,
            input_names = ['_', 'subject_id', 'run_id', 'working_dir']),
            name = 'remove_smoothed_files')
        remove_smoothed_files.inputs.working_dir = self.directories.working_dir

        # Create l1 analysis workflow and connect its nodes
        run_level_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = 'run_level_analysis'
            )
        run_level_analysis.connect([
            (information_source, select_files, [
                ('subject_id', 'subject_id'),
                ('run_id', 'run_id')]),
            (select_files, subject_information, [('event', 'event_file')]),
            (select_files, parameters, [('param', 'filepath')]),
            (information_source, parameters, [
                ('subject_id', 'subject_id'),
                ('run_id', 'run_id')]),
            (select_files, skull_stripping_func, [('func', 'in_file')]),
            (skull_stripping_func, smoothing_func, [('out_file', 'in_file')]),
            (parameters, specify_model, [('parameters_file', 'realignment_parameters')]),
            (smoothing_func, specify_model, [('out_file', 'functional_runs')]),
            (subject_information, specify_model, [('subject_info', 'subject_info')]),
            (contrasts, model_design, [('contrasts', 'contrasts')]),
            (specify_model, model_design, [('session_info', 'session_info')]),
            (model_design, model_generation, [
                ('ev_files', 'ev_files'),
                ('fsf_files', 'fsf_file')]),
            (smoothing_func, model_estimate, [('out_file', 'in_file')]),
            (model_generation, model_estimate, [
                ('con_file', 'tcon_file'),
                ('design_file', 'design_file')]),
            (information_source, remove_smoothed_files, [
                ('subject_id', 'subject_id'),
                ('run_id', 'run_id')]),
            (model_estimate, remove_smoothed_files, [('results_dir', '_')]),
            (model_estimate, datasink, [('results_dir', 'run_level_analysis.@results')]),
            (model_generation, datasink, [
                ('design_file', 'run_level_analysis.@design_file'),
                ('design_image', 'run_level_analysis.@design_img')]),
            (skull_stripping_func, datasink, [('mask_file', 'run_level_analysis.@skullstriped')])
            ])

        return run_level_analysis

    def get_run_level_outputs(self):
        """ Return the names of the files the run level analysis is supposed to generate. """

        parameters = {
            'run_id' : self.run_list,
            'subject_id' : self.subject_list,
            'file' : [
                join('results', 'dof'),
                join('results', 'logfile'),
                join('results', 'pe10.nii.gz'),
                join('results', 'pe11.nii.gz'),
                join('results', 'pe12.nii.gz'),
                join('results', 'pe13.nii.gz'),
                join('results', 'pe14.nii.gz'),
                join('results', 'pe15.nii.gz'),
                join('results', 'pe16.nii.gz'),
                join('results', 'pe17.nii.gz'),
                join('results', 'pe1.nii.gz'),
                join('results', 'pe2.nii.gz'),
                join('results', 'pe3.nii.gz'),
                join('results', 'pe4.nii.gz'),
                join('results', 'pe5.nii.gz'),
                join('results', 'pe6.nii.gz'),
                join('results', 'pe7.nii.gz'),
                join('results', 'pe8.nii.gz'),
                join('results', 'pe9.nii.gz'),
                join('results', 'res4d.nii.gz'),
                join('results', 'sigmasquareds.nii.gz'),
                join('results', 'threshac1.nii.gz'),
                'run0.mat',
                'run0.png',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_space-MNI152NLin2009cAsym_preproc_brain_mask.nii.gz'
            ]
        }
        parameter_sets = product(*parameters.values())
        template = join(
            self.directories.output_dir,
            'run_level_analysis', '_run_id_{run_id}_subject_id_{subject_id}','{file}'
            )
        return_list = [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

        parameters = {
            'run_id' : self.run_list,
            'subject_id' : self.subject_list,
            'contrast_id' : self.contrast_list,
            'file' : [
                join('results', 'cope{contrast_id}.nii.gz'),
                join('results', 'tstat{contrast_id}.nii.gz'),
                join('results', 'varcope{contrast_id}.nii.gz'),
                join('results', 'zstat{contrast_id}.nii.gz'),
            ]
        }
        parameter_sets = product(*parameters.values())
        template = join(
            self.directories.output_dir,
            'run_level_analysis', '_run_id_{run_id}_subject_id_{subject_id}','{file}'
            )

        return_list += [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

        return return_list

    def get_subject_level_analysis(self):
        """
        Create the subject level analysis workflow.

        Returns:
        - subject_level_analysis : nipype.WorkFlow
        """
        # Infosource Node - To iterate on subject and runs
        information_source = Node(IdentityInterface(
            fields = ['subject_id', 'contrast_id']),
            name = 'information_source')
        information_source.iterables = [
            ('subject_id', self.subject_list),
            ('contrast_id', self.contrast_list)
            ]

        # Templates to select files node
        templates = {
            'cope' : join(self.directories.output_dir,
                'run_level_analysis', '_run_id_*_subject_id_{subject_id}', 'results',
                'cope{contrast_id}.nii.gz'),
            'varcope' : join(self.directories.output_dir,
                'run_level_analysis', '_run_id_*_subject_id_{subject_id}', 'results',
                'varcope{contrast_id}.nii.gz'),
            'mask': join(self.directories.output_dir,
                'run_level_analysis', '_run_id_*_subject_id_{subject_id}',
                'sub-{subject_id}_task-MGT_run-*_bold_space-MNI152NLin2009cAsym_preproc_brain_mask.nii.gz')
        }

        # SelectFiles Node - to select necessary files
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.results_dir

        # DataSink Node - store the wanted results in the wanted directory
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir

        # L2Model Node - Generate subject specific second level model
        generate_model = Node(L2Model(), name = 'generate_model')
        generate_model.inputs.num_copes = len(self.run_list)

        # Merge Node - Merge copes files for each subject
        merge_copes = Node(Merge(), name = 'merge_copes')
        merge_copes.inputs.dimension = 't'

        # Merge Node - Merge varcopes files for each subject
        merge_varcopes = Node(Merge(), name = 'merge_varcopes')
        merge_varcopes.inputs.dimension = 't'

        # FLAMEO Node - Estimate model
        estimate_model = Node(FLAMEO(), name = 'estimate_model')
        estimate_model.inputs.run_mode = 'flame1'

        # Second level (single-subject, mean of all four scans) analyses: Fixed effects analysis.
        subject_level_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = 'subject_level_analysis')
        subject_level_analysis.connect([
            (information_source, select_files, [
                ('subject_id', 'subject_id'),
                ('contrast_id', 'contrast_id')]),
            (select_files, merge_copes, [('cope', 'in_files')]),
            (select_files, merge_varcopes, [('varcope', 'in_files')]),
            (select_files, estimate_model, [('mask', 'mask_file')]),
            (merge_copes, estimate_model, [('merged_file', 'cope_file')]),
            (merge_varcopes, estimate_model, [('merged_file', 'var_cope_file')]),
            (generate_model, estimate_model, [
                ('design_mat', 'design_file'),
                ('design_con', 't_con_file'),
                ('design_grp', 'cov_split_file')]),
            (estimate_model, data_sink, [
                ('zstats', 'subject_level_analysis.@stats'),
                ('tstats', 'subject_level_analysis.@tstats'),
                ('copes', 'subject_level_analysis.@copes'),
                ('var_copes', 'subject_level_analysis.@varcopes')])])

        return subject_level_analysis

    def get_subject_level_outputs(self):
        """ Return the names of the files the subject level analysis is supposed to generate. """

        parameters = {
            'contrast_id' : self.contrast_list,
            'subject_id' : self.subject_list,
            'file' : ['cope1.nii.gz', 'tstat1.nii.gz', 'varcope1.nii.gz', 'zstat1.nii.gz']
        }
        parameter_sets = product(*parameters.values())
        template = join(
            self.directories.output_dir,
            'subject_level_analysis', '_contrast_id_{contrast_id}_subject_id_{subject_id}','{file}'
            )

        return [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

    def get_subgroups_contrasts(copes, varcopes, subject_list: list, participants_file: str):
        """
        Return the file list containing only the files belonging to subject in the wanted group.

        Parameters :
        - copes: original file list selected by select_files node
        - varcopes: original file list selected by select_files node
        - subject_list: list of subject IDs that are analyzed
        - participants_file: file containing participants characteristics

        Returns :
        - copes_equal_indifference : a subset of copes corresponding to subjects
          in the equalIndifference group
        - copes_equal_range : a subset of copes corresponding to subjects
          in the equalRange group
        - varcopes_equal_indifference : a subset of varcopes corresponding to subjects
          in the equalIndifference group
        - varcopes_equal_range : a subset of varcopes corresponding to subjects
          in the equalRange group
        - equal_indifference_ids : a list of subject ids in the equalIndifference group
        - equal_range_ids : a list of subject ids in the equalRange group
        """

        subject_list_sub_ids = [] # ids as written in the participants file
        equal_range_ids = [] # ids as 3-digit string
        equal_indifference_ids = [] # ids as 3-digit string
        equal_range_sub_ids = [] # ids as written in the participants file
        equal_indifference_sub_ids = [] # ids as written in the participants file

        # Reading file containing participants IDs and groups
        with open(participants_file, 'rt') as file:
            next(file)  # skip the header

            for line in file:
                info = line.strip().split()
                subject_id = info[0][-3:]
                subject_group = info[1]

                # Check if the participant ID was selected and sort depending on group
                if subject_id in subject_list:
                    subject_list_sub_ids.append(info[0])
                    if subject_group == 'equalIndifference':
                        equal_indifference_ids.append(subject_id)
                        equal_indifference_sub_ids.append(info[0])
                    elif subject_group == 'equalRange':
                        equal_range_ids.append(subject_id)
                        equal_range_sub_ids.append(info[0])


        # Return sorted selected copes and varcopes by group, and corresponding ids
        return \
            [c for c in copes if any(i in c for i in equal_indifference_sub_ids)],\
            [c for c in copes if any(i in c for i in equal_range_sub_ids)],\
            [c for c in copes if any(i in c for i in subject_list_sub_ids)],\
            [v for v in varcopes if any(i in v for i in equal_indifference_sub_ids)],\
            [v for v in varcopes if any(i in v for i in equal_range_sub_ids)],\
            [v for v in varcopes if any(i in v for i in subject_list_sub_ids)],\
            equal_indifference_ids, equal_range_ids

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
        equal_range_ids: list,
        equal_indifference_ids: list,
        subject_list: list,
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
            - group_level_analysis: nipype.WorkFlow
        """
        # Infosource Node - iterate over the contrasts generated by the subject level analysis
        information_source = Node(IdentityInterface(
            fields = ['contrast_ids', 'subject_list'],
            subject_list = self.subject_list),
            name = 'information_source')
        information_source.iterables = [('contrast_id', self.contrast_list)]

        # SelectFiles Node - select necessary files
        templates = {
            'cope' : join(self.directories.output_dir,
                'l2_analysis', '_contrast_id_{contrast_id}_subject_id_{subject_id}',
                'cope1.nii.gz'),
            'varcope' : join(self.directories.output_dir,
                'l2_analysis', '_contrast_id_{contrast_id}_subject_id_{subject_id}',
                'varcope1.nii.gz'),
            'participants' : join(self.directories.dataset_dir, 'participants.tsv'),
            'mask': join(self.directories.output_dir,
                'l1_analysis', '_run_id_*_subject_id_{subject_id}',
                'sub-{subject_id}_task-MGT_run-*_bold_space-MNI152NLin2009cAsym_preproc_brain_mask.nii.gz')
            }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.results_dir

        # Datasink Node - save important files
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir

        # Merge Node - Merge cope files
        merge_copes = Node(Merge(), name = 'merge_copes')
        merge_copes.inputs.dimension = 't'

        # Merge Node - Merge cope files
        merge_varcopes = Node(Merge(), name = 'merge_varcopes')
        merge_varcopes.inputs.dimension = 't'

        # Function Node get_one_sample_t_test_regressors
        #   Get regressors in the equalRange and equalIndifference method case
        regressors_one_sample = Node(
            Function(
                function = self.get_one_sample_t_test_regressors,
                input_names = ['subject_list'],
                output_names = ['regressors']
            ),
            name = 'regressors_one_sample',
        )

        # Function Node get_two_sample_t_test_regressors
        #   Get regressors in the groupComp method case
        regressors_two_sample = Node(
            Function(
                function = self.get_two_sample_t_test_regressors,
                input_names = [
                    'equal_range_ids',
                    'equal_indifference_ids',
                    'subject_list',
                ],
                output_names = ['regressors', 'groups']
            ),
            name = 'regressors_two_sample',
        )
        regressors_two_sample.inputs.subject_list = self.subject_list

        # Function Node get_subgroups_contrasts - Get the contrast files for each subgroup
        get_contrasts = Node(Function(
            function = self.get_subgroups_contrasts,
            input_names = ['copes', 'varcopes', 'subject_list', 'participants_file'],
            output_names = [
                'copes_equal_indifference',
                'copes_equal_range',
                'copes_global',
                'varcopes_equal_indifference',
                'varcopes_equal_range',
                'varcopes_global',
                'equal_indifference_id',
                'equal_range_id'
                ]
            ),
            name = 'get_contrasts')

        # MultipleRegressDesign Node - Specify model
        specify_model = Node(MultipleRegressDesign(), name = 'specify_model')

        # FLAMEO Node - Estimate model
        estimate_model = Node(FLAMEO(), name = 'estimate_model')
        estimate_model.inputs.run_mode = 'flame1'

        # Randomise Node -
        randomise = Node(Randomise(), name = 'randomise')
        randomise.inputs.num_perm = 10000
        randomise.inputs.tfce = True
        randomise.inputs.vox_p_values = True
        randomise.inputs.c_thresh = 0.05
        randomise.inputs.tfce_E = 0.01

        # Compute the number of participants used to do the analysis
        nb_subjects = len(self.subject_list)

        # Declare the workflow
        group_level_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = f'group_level_analysis_{method}_nsub_{nb_subjects}')
        group_level_analysis.connect([
            (information_source, select_files, [('contrast_id', 'contrast_id')]),
            (information_source, get_contrasts, [('subject_list', 'subject_list')]),
            (select_files, get_contrasts, [
                ('cope', 'copes'),
                ('varcope', 'varcopes'),
                ('participants', 'participants_file')]),
            (select_files, estimate_model, [('mask', 'mask_file')]),
            (select_files, randomise, [('mask', 'mask')])
            ])

        if method in ('equalIndifference', 'equalRange'):
            specify_model.inputs.contrasts = [
                ['group_mean', 'T', ['group_mean'], [1]],
                ['group_mean_neg', 'T', ['group_mean'], [-1]]
                ]

            group_level_analysis.connect([
                (regressors_one_sample, specify_model, [('regressors', 'regressors')])
                ])

            if method == 'equalIndifference':
                group_level_analysis.connect([
                    (get_contrasts, merge_copes, [('copes_equal_indifference', 'in_files')]),
                    (get_contrasts, merge_varcopes,[('varcopes_equal_indifference', 'in_files')]),
                    (get_contrasts, regressors_one_sample, [
                        ('equal_indifference_id', 'subject_list')
                        ])
                ])

            elif method == 'equalRange':
                group_level_analysis.connect([
                    (get_contrasts, merge_copes, [('copes_equal_range', 'in_files')]),
                    (get_contrasts, merge_varcopes, [('varcopes_equal_range', 'in_files')]),
                    (get_contrasts, regressors_one_sample, [('equal_range_id', 'equal_range_id')])
                ])

        elif method == 'groupComp':
            specify_model.inputs.contrasts = [
                ['equalRange_sup', 'T', ['equalRange', 'equalIndifference'], [1, -1]]
            ]
            group_level_analysis.connect([
                (get_contrasts, merge_copes, [('copes_global', 'in_files')]),
                (get_contrasts, merge_varcopes,[('varcopes_global', 'in_files')]),
                (get_contrasts, regressors_two_sample, [
                    ('equal_range_id', 'equal_range_id'),
                    ('equal_indifference_id', 'equal_indifference_id')]),
                (regressors_two_sample, specify_model, [
                    ('regressors', 'regressors'),
                    ('groups', 'groups')])
                ])

        group_level_analysis.connect([
            (merge_copes, estimate_model, [('merged_file', 'cope_file')]),
            (merge_varcopes, estimate_model, [('merged_file', 'var_cope_file')]),
            (specify_model, estimate_model, [
                ('design_mat', 'design_file'),
                ('design_con', 't_con_file'),
                ('design_grp', 'cov_split_file')]),
            (merge_copes, randomise, [('merged_file', 'in_file')]),
            (specify_model, randomise, [
                ('design_mat', 'design_mat'),
                ('design_con', 'tcon')]),
            (randomise, data_sink, [
                ('t_corrected_p_files', f'group_level_analysis_{method}_nsub_{nb_subjects}.@tcorpfile'),
                ('tstat_files', f'group_level_analysis_{method}_nsub_{nb_subjects}.@tstat')]),
            (estimate_model, data_sink, [
                ('zstats', f'group_level_analysis_{method}_nsub_{nb_subjects}.@zstats'),
                ('tstats', f'group_level_analysis_{method}_nsub_{nb_subjects}.@tstats')]),
        ])

        return group_level_analysis

    def get_group_level_outputs(self):
        """ Return all names for the files the group level analysis is supposed to generate. """

        # Handle equalRange and equalIndifference
        parameters = {
            'contrast_id': ['ploss', 'pgain'],
            'method': ['equalRange', 'equalIndifference'],
            'file': [
                'randomise_tfce_corrp_tstat1.nii.gz',
                'randomise_tfce_corrp_tstat2.nii.gz',
                'randomise_tstat1.nii.gz',
                'randomise_tstat2.nii.gz',
                'tstat1.nii.gz',
                'tstat2.nii.gz',
                'zstat1.nii.gz',
                'zstat2.nii.gz'
                ],
            'nb_subjects' : [str(len(self.subject_list))]
        }
        parameter_sets = product(*parameters.values())
        template = join(
            self.directories.output_dir,
            'l3_analysis_{method}_nsub_{nb_subjects}',
            '_contrast_id_{contrast_id}',
            '{file}'
            )

        return_list = [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

        # Handle groupComp
        files = [
            'randomise_tfce_corrp_tstat1.nii.gz',
            'randomise_tstat1.nii.gz',
            'zstat1.nii.gz',
            'tstat1.nii.gz'
            ]

        return_list += [join(
            self.directories.output_dir,
            f'l3_analysis_groupComp_nsub_{len(self.subject_list)}',
            '_contrast_id_ploss', f'{file}') for file in files]

        return return_list

    def get_hypotheses_outputs(self):
        """ Return all hypotheses output file names. """

        nb_sub = len(self.subject_list)
        files = [
            join(f'l3_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_pgain', 'randomise_tfce_corrp_tstat1.nii.gz'),
            join(f'l3_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_pgain', 'zstat1.nii.gz'),
            join(f'l3_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_pgain', 'randomise_tfce_corrp_tstat1.nii.gz'),
            join(f'l3_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_pgain', 'zstat1.nii.gz'),
            join(f'l3_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_pgain', 'randomise_tfce_corrp_tstat1.nii.gz'),
            join(f'l3_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_pgain', 'zstat1.nii.gz'),
            join(f'l3_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_pgain', 'randomise_tfce_corrp_tstat1.nii.gz'),
            join(f'l3_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_pgain', 'zstat1.nii.gz'),
            join(f'l3_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_ploss', 'randomise_tfce_corrp_tstat2.nii.gz'),
            join(f'l3_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_ploss', 'zstat2.nii.gz'),
            join(f'l3_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_ploss', 'randomise_tfce_corrp_tstat2.nii.gz'),
            join(f'l3_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_ploss', 'zstat2.nii.gz'),
            join(f'l3_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_ploss', 'randomise_tfce_corrp_tstat1.nii.gz'),
            join(f'l3_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_ploss', 'zstat1.nii.gz'),
            join(f'l3_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_ploss', 'randomise_tfce_corrp_tstat1.nii.gz'),
            join(f'l3_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_ploss', 'zstat1.nii.gz'),
            join(f'l3_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_ploss', 'randomise_tfce_corrp_tstat1.nii.gz'),
            join(f'l3_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_ploss', 'zstat1.nii.gz')
        ]
        return [join(self.directories.output_dir, f) for f in files]

##### TODO : what is this ?
#system('export PATH=$PATH:/local/egermani/ICA-AROMA')
