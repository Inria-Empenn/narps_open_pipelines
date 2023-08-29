#!/usr/bin/python
# coding: utf-8

""" Write the work of NARPS team T54A using Nipype """

from os import system
from os.path import join
from itertools import product

from nipype import Workflow, Node
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.fsl import (
    BET, IsotropicSmooth, Level1Design, FEATModel, L2Model, Merge, FLAMEO,
    FILMGLS, Randomise, MultipleRegressDesign
    )
from nipype.algorithms.modelgen import SpecifyModel

from narps_open.pipelines import Pipeline

class PipelineTeamT54A(Pipeline):
    """ A class that defines the pipeline of team T54A. """

    def __init__(self):
        super().__init__()
        self.fwhm = 4.0
        self.team_id = 'T54A'
        self.contrast_list = ['01', '02']

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
                        elif condition == 'difficonditionulty':
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
        from os import mkdir
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

        if not isdir(join(working_dir, 'parameters_file')):
            mkdir(join(working_dir, 'parameters_file'))

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
            - l1_analysis : nipype.WorkFlow
        """
        # Infosource Node - To iterate on subject and runs
        infosource = Node(IdentityInterface(
            fields = ['subject_id', 'run_id']),
            name = 'infosource')
        infosource.iterables = [
            ('subject_id', self.subject_list),
            ('run_id', self.run_list)]

        # Templates to select files node
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

        # SelectFiles - to select necessary files
        selectfiles = Node(SelectFiles(template, base_directory = self.directories.dataset_dir),
            name = 'selectfiles')

        # DataSink - store the wanted results in the wanted repository
        datasink = Node(DataSink(base_directory = self.directories.output_dir),
            name='datasink')

        # Skullstripping
        skullstrip = Node(BET(frac = 0.1, functional = True, mask = True),
            name = 'skullstrip')

        # Smoothing
        smooth = Node(IsotropicSmooth(fwhm = self.fwhm),
            name = 'smooth')

        # Function node get_subject_infos - get subject specific condition information
        subject_infos = Node(Function(
            function = self.get_session_infos,
            input_names = ['event_file'],
            output_names = ['subject_info']),
            name = 'subject_infos')

        # SpecifyModel - generates Model
        specify_model = Node(SpecifyModel(
            high_pass_filter_cutoff = 100, input_units = 'secs', time_repetition = self.tr),
            name = 'specify_model')

        # Node contrasts to get contrasts
        contrasts = Node(Function(
            function = self.get_contrasts,
            input_names = ['subject_id'],
            output_names = ['contrasts']),
            name = 'contrasts')

        # Function node get_parameters_file - get parameters files
        parameters = Node(Function(
            function = self.get_parameters_file,
            input_names = ['filepath', 'subject_id', 'run_id', 'working_dir'],
            output_names=['parameters_file']),
            name='parameters')
        parameters.inputs.working_dir = self.directories.working_dir

        # First temporal derivatives of the two regressors were also used,
        # along with temporal filtering (60 s) of all the independent variable time-series.
        # No motion parameter regressors used.
        # Level1Design - generates a design matrix
        l1_design = Node(Level1Design(
            bases = {'dgamma':{'derivs' : True}},
            interscan_interval = self.tr,
            model_serial_correlations = True),
            name = 'l1_design')

        model_generation = Node(FEATModel(),
            name = 'model_generation')

        model_estimate = Node(FILMGLS(),
            name='model_estimate')

        # Function node remove_smoothed_files - remove output of the smooth node
        remove_smoothed_files = Node(Function(
            function = self.remove_smoothed_files,
            input_names = ['_', 'subject_id', 'run_id', 'working_dir']),
            name = 'remove_smoothed_files')
        remove_smoothed_files.inputs.working_dir = self.directories.working_dir

        # Create l1 analysis workflow and connect its nodes
        l1_analysis = Workflow(base_dir = self.directories.working_dir, name = 'l1_analysis')
        l1_analysis.connect([
            (infosource, selectfiles, [
                ('subject_id', 'subject_id'),
                ('run_id', 'run_id')]),
            (selectfiles, subject_infos, [('event', 'event_file')]),
            (selectfiles, parameters, [('param', 'filepath')]),
            (infosource, contrasts, [('subject_id', 'subject_id')]),
            (infosource, parameters, [
                ('subject_id', 'subject_id'),
                ('run_id', 'run_id')]),
            (selectfiles, skullstrip, [('func', 'in_file')]),
            (skullstrip, smooth, [('out_file', 'in_file')]),
            (parameters, specify_model, [('parameters_file', 'realignment_parameters')]),
            (smooth, specify_model, [('out_file', 'functional_runs')]),
            (subject_infos, specify_model, [('subject_info', 'subject_info')]),
            (contrasts, l1_design, [('contrasts', 'contrasts')]),
            (specify_model, l1_design, [('session_info', 'session_info')]),
            (l1_design, model_generation, [
                ('ev_files', 'ev_files'),
                ('fsf_files', 'fsf_file')]),
            (smooth, model_estimate, [('out_file', 'in_file')]),
            (model_generation, model_estimate, [
                ('con_file', 'tcon_file'),
                ('design_file', 'design_file')]),
            (infosource, remove_smoothed_files, [
                ('subject_id', 'subject_id'),
                ('run_id', 'run_id')]),
            (model_estimate, remove_smoothed_files, [('results_dir', '_')]),
            (model_estimate, datasink, [('results_dir', 'l1_analysis.@results')]),
            (model_generation, datasink, [
                ('design_file', 'l1_analysis.@design_file'),
                ('design_image', 'l1_analysis.@design_img')]),
            (skullstrip, datasink, [('mask_file', 'l1_analysis.@skullstriped')])
            ])

        return l1_analysis

    def get_run_level_outputs(self):
        """ Return the names of the files the run level analysis is supposed to generate. """

        parameters = {
            'run_id' : self.run_list,
            'subject_id' : self.subject_list,
            'file' : [
                join('results', 'cope1.nii.gz'),
                join('results', 'cope2.nii.gz'),
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
                join('results', 'tstat1.nii.gz'),
                join('results', 'tstat2.nii.gz'),
                join('results', 'varcope1.nii.gz'),
                join('results', 'varcope2.nii.gz'),
                join('results', 'zstat1.nii.gz'),
                join('results', 'zstat2.nii.gz'),
                'run0.mat',
                'run0.png',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_space-MNI152NLin2009cAsym_preproc_brain_mask.nii.gz'
            ]
        }
        parameter_sets = product(*parameters.values())
        template = join(
            self.directories.output_dir,
            'l1_analysis', '_run_id_{run_id}_subject_id_{subject_id}','{file}'
            )

        return [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

    def get_subject_level_analysis(self):
        """
        Create the subject level analysis workflow.

        Returns:
        - l2_analysis: nipype.WorkFlow
        """
        # Infosource Node - To iterate on subject and runs
        infosource_sub_level = Node(IdentityInterface(
            fields = ['subject_id', 'contrast_id']),
            name = 'infosource_sub_level')
        infosource_sub_level.iterables = [
            ('subject_id', self.subject_list),
            ('contrast_id', self.contrast_list)
            ]

        # Templates to select files node
        templates = {
            'cope' : join(self.directories.output_dir,
                'l1_analysis', '_run_id_*_subject_id_{subject_id}', 'results',
                'cope{contrast_id}.nii.gz'),
            'varcope' : join(self.directories.output_dir,
                'l1_analysis', '_run_id_*_subject_id_{subject_id}', 'results',
                'varcope{contrast_id}.nii.gz'),

            ##### TODO not dataset here
            'mask': join(self.directories.dataset_dir, 'NARPS-T54A', 'hypo1_cope.nii.gz')
        }

        # SelectFiles node - to select necessary files
        selectfiles_sub_level = Node(SelectFiles(
            templates, base_directory = self.directories.results_dir),
            name = 'selectfiles_sub_level')

        # Datasink - save important files
        datasink_sub_level = Node(DataSink(
            base_directory = str(self.directories.output_dir)
            ),
            name = 'datasink_sub_level')

        # Generate design matrix
        specify_model_sub_level = Node(L2Model(
            num_copes = len(self.run_list)),
            name = 'specify_model_sub_level')

        # Merge copes and varcopes files for each subject
        merge_copes = Node(Merge(dimension = 't'),
            name='merge_copes')
        merge_varcopes = Node(Merge(dimension='t'),
            name='merge_varcopes')

        flame = Node(FLAMEO(run_mode = 'flame1'),
            name='flameo')

        # Second level (single-subject, mean of all four scans) analyses: Fixed effects analysis.
        l2_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = 'l2_analysis')
        l2_analysis.connect([
            (infosource_sub_level, selectfiles_sub_level, [
                ('subject_id', 'subject_id'),
                ('contrast_id', 'contrast_id')]),
            (selectfiles_sub_level, merge_copes, [('cope', 'in_files')]),
            (selectfiles_sub_level, merge_varcopes, [('varcope', 'in_files')]),
            (selectfiles_sub_level, flame, [('mask', 'mask_file')]),
            (merge_copes, flame, [('merged_file', 'cope_file')]),
            (merge_varcopes, flame, [('merged_file', 'var_cope_file')]),
            (specify_model_sub_level, flame, [
                ('design_mat', 'design_file'),
                ('design_con', 't_con_file'),
                ('design_grp', 'cov_split_file')]),
            (flame, datasink_sub_level, [
                ('zstats', 'l2_analysis.@stats'),
                ('tstats', 'l2_analysis.@tstats'),
                ('copes', 'l2_analysis.@copes'),
                ('var_copes', 'l2_analysis.@varcopes')])])

        return l2_analysis

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
            'l2_analysis', '_contrast_id_{contrast_id}_subject_id_{subject_id}','{file}'
            )

        return [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

    def get_subgroups_contrasts(copes, varcopes, subject_ids, participants_file):
        """
        Parameters :
        - copes: original file list selected by selectfiles node
        - varcopes: original file list selected by selectfiles node
        - subject_ids: list of subject IDs that are analyzed
        - participants_file: str, file containing participants characteristics

        Returns : the file list containing only the files belonging to subject in the wanted group.
        """
        equal_range_id = []
        equal_indifference_id = []
        subject_list = [f'sub-{str(s).zfill(3)}' for s in subject_ids]

        with open(participants_file, 'rt') as file:
            next(file)  # skip the header

            for line in file:
                info = line.strip().split()
                if info[0] in subject_list and info[1] == 'equalIndifference':
                    equal_indifference_id.append(info[0][-3:])
                elif info[0] in subject_list and info[1] == 'equalRange':
                    equal_range_id.append(info[0][-3:])

        copes_equal_indifference = []
        copes_equal_range = []
        copes_global = []
        varcopes_equal_indifference = []
        varcopes_equal_range = []
        varcopes_global = []

        for file in copes:
            sub_id = file.split('/')
            if sub_id[-1][4:7] in equal_indifference_id:
                copes_equal_indifference.append(file)
            elif sub_id[-1][4:7] in equal_range_id:
                copes_equal_range.append(file)
            if sub_id[-1][4:7] in subject_ids:
                copes_global.append(file)

        for file in varcopes:
            sub_id = file.split('/')
            if sub_id[-1][4:7] in equal_indifference_id:
                varcopes_equal_indifference.append(file)
            elif sub_id[-1][4:7] in equal_range_id:
                varcopes_equal_range.append(file)
            if sub_id[-1][4:7] in subject_ids:
                varcopes_global.append(file)

        return copes_equal_indifference, copes_equal_range,\
            varcopes_equal_indifference, varcopes_equal_range,\
            equal_indifference_id, equal_range_id,\
            copes_global, varcopes_global

    def get_regressors(equal_range_ids, equal_indifference_ids, method, subject_list):
        """
        Create dictionary of regressors for group analysis.

        Parameters:
            - equal_range_ids: list of str, ids of subjects in equal range group
            - equal_indifference_ids: list of str, ids of subjects in equal indifference group
            - method: one of 'equalRange', 'equalIndifference' or 'groupComp'
            - subject_list: list of str, ids of subject for which to do the analysis

        Returns:
            - regressors: dict, dictionary of regressors used to
                distinguish groups in FSL group analysis
        """
        if method == 'equalRange':
            regressors = dict(group_mean = [1 for i in range(len(equal_range_ids))])
        elif method == 'equalIndifference':
            regressors = dict(group_mean = [1 for i in range(len(equal_indifference_ids))])
        elif method == 'groupComp':
            equal_range_reg = [
                1 for i in range(len(equal_range_ids) + len(equal_indifference_ids))
                ]
            equal_indifference_reg = [
                0 for i in range(len(equal_range_ids) + len(equal_indifference_ids))
                ]

            for index, sub_id in enumerate(subject_list):
                if sub_id in equal_indifference_ids:
                    equal_indifference_reg[index] = 1
                    equal_range_reg[index] = 0

            regressors = dict(
                equalRange = equal_range_reg,
                equalIndifference = equal_indifference_reg
                )

        return regressors

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
            - l3_analysis: nipype.WorkFlow
        """
        # Compute the number of participants used to do the analysis
        nb_subjects = len(self.subject_list)

        # Infosource Node - To iterate on subject and runs
        infosource_group_level = Node(IdentityInterface(
            fields = ['contrast_ids', 'subject_list'],
            subject_list = self.subject_list),
            name = 'infosource_group_level')
        infosource_group_level.iterables = [('contrast_id', self.contrast_list)]

        # Templates to select files node
        templates = {
            ##### TODO not dataset here
            'cope' : join(self.directories.output_dir,
                'l2_analysis', '_contrast-{contrast_id}_subject_id_cope.nii.gz'),
            ##### TODO not dataset here
            'varcope' : join(data_dir, 'NARPS-T54A', 'sub-*_contrast-{contrast_id}_varcope.nii.gz'),
            'participants' : join(self.directories.dataset_dir, 'participants.tsv'),
            ##### TODO not dataset here
            'mask' : join(data_dir, 'NARPS-T54A', 'hypo2_unthresh_Z.nii.gz')
            }

        # SelectFiles node - to select necessary files
        selectfiles_group_level = Node(SelectFiles(
            templates, base_directory = self.directories.result_dir),
            name = 'selectfiles_group_level')

        datasink_group_level = Node(DataSink(
            base_directory = self.directories.result_dir
            ),
            name='datasink_group_level')

        merge_copes_group_level = Node(Merge(dimension = 't'),
            name = 'merge_copes_group_level')
        merge_varcopes_group_level = Node(Merge(dimension = 't'),
            name = 'merge_varcopes_group_level')

        subgroups_contrasts = Node(Function(
            function = self.get_subgroups_contrasts,
            input_names = ['copes', 'varcopes', 'subject_list', 'participants_file'],
            output_names = [
                'copes_equalIndifference', 'copes_equalRange',
                'varcopes_equalIndifference', 'varcopes_equalRange',
                'equalIndifference_id', 'equalRange_id',
                'copes_global', 'varcopes_global']
            ),
            name = 'subgroups_contrasts')

        specifymodel_group_level = Node(MultipleRegressDesign(),
            name = 'specifymodel_group_level')

        flame_group_level = Node(FLAMEO(run_mode = 'flame1'),
            name='flame_group_level')

        regressors = Node(Function(
            function = self.get_regressors,
            input_names = ['equalRange_id', 'equalIndifference_id', 'method', 'subject_list'],
            output_names = ['regressors']),
            name = 'regressors')
        regressors.inputs.method = method
        regressors.inputs.subject_list = self.subject_list

        randomise = Node(Randomise(
            num_perm = 10000, tfce = True, vox_p_values = True, c_thresh = 0.05, tfce_E = 0.01),
            name = "randomise")

        l3_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = f'l3_analysis_{method}_nsub_{nb_subjects}')
        l3_analysis.connect([
            (infosource_group_level, selectfiles_group_level, [('contrast_id', 'contrast_id')]),
            (infosource_group_level, subgroups_contrasts, [('subject_list', 'subject_ids')]),
            (selectfiles_group_level, subgroups_contrasts, [
                ('cope', 'copes'),
                ('varcope', 'varcopes'),
                ('participants', 'participants_file')]),
            (selectfiles_group_level, flame_group_level, [('mask', 'mask_file')]),
            (selectfiles_group_level, randomise, [('mask', 'mask')]),
            (subgroups_contrasts, regressors, [
                ('equalRange_id', 'equalRange_id'),
                ('equalIndifference_id', 'equalIndifference_id')]),
            (regressors, specifymodel_group_level, [('regressors', 'regressors')])])

        if method in ('equalIndifference', 'equalRange'):
            specifymodel_group_level.inputs.contrasts = [
                ['group_mean', 'T', ['group_mean'], [1]],
                ['group_mean_neg', 'T', ['group_mean'], [-1]]
                ]

            if method == 'equalIndifference':
                l3_analysis.connect([
                    (subgroups_contrasts, merge_copes_group_level,
                        [('copes_equalIndifference', 'in_files')]
                    ),
                    (subgroups_contrasts, merge_varcopes_group_level,
                        [('varcopes_equalIndifference', 'in_files')]
                    )
                ])
            elif method == 'equalRange':
                l3_analysis.connect([
                    (subgroups_contrasts, merge_copes_group_level,
                        [('copes_equalRange', 'in_files')]
                    ),
                    (subgroups_contrasts, merge_varcopes_group_level,
                        [('varcopes_equalRange', 'in_files')]
                    )
                ])

        elif method == 'groupComp':
            specifymodel_group_level.inputs.contrasts = [
                ['equalRange_sup', 'T', ['equalRange', 'equalIndifference'], [1, -1]]
            ]
            l3_analysis.connect([
                (subgroups_contrasts, merge_copes_group_level,
                    [('copes_global', 'in_files')]
                ),
                (subgroups_contrasts, merge_varcopes_group_level,
                    [('varcopes_global', 'in_files')]
                )
            ])

        l3_analysis.connect([
            (merge_copes_group_level, flame_group_level, [('merged_file', 'cope_file')]),
            (merge_varcopes_group_level, flame_group_level, [('merged_file', 'var_cope_file')]),
            (specifymodel_group_level, flame_group_level, [
                ('design_mat', 'design_file'),
                ('design_con', 't_con_file'),
                ('design_grp', 'cov_split_file')]),
            (merge_copes_group_level, randomise, [('merged_file', 'in_file')]),
            (specifymodel_group_level, randomise, [
                ('design_mat', 'design_mat'),
                ('design_con', 'tcon')]),
            (randomise, datasink_group_level, [
                ('t_corrected_p_files', f'l3_analysis_{method}_nsub_{nb_subjects}.@tcorpfile'),
                ('tstat_files', f'l3_analysis_{method}_nsub_{nb_subjects}.@tstat')]),
            (flame_group_level, datasink_group_level, [
                ('zstats', f'l3_analysis_{method}_nsub_{nb_subjects}.@zstats'),
                ('tstats', f'l3_analysis_{method}_nsub_{nb_subjects}.@tstats')]),
        ])

        return l3_analysis

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
        files : [
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
system('export PATH=$PATH:/local/egermani/ICA-AROMA')
