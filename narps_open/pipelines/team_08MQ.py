#!/usr/bin/python
# coding: utf-8

""" Write the work of NARPS team 08MQ using Nipype """

"""
This template can be use to reproduce a pipeline using FSL as main software.

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
# from nipype.algorithms.misc import Gunzip

# from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces.fsl import (
    FAST, BET, Registration, ErodeImage, PrepareFieldmap, MCFLIRT, SliceTimer
    )


"""
    Info, ImageMaths, IsotropicSmooth, Threshold,
    Level1Design, FEATModel, L2Model, Merge,
    FLAMEO, ContrastMgr, FILMGLS, MultipleRegressDesign,
    Cluster, BET, SmoothEstimate
    )
"""

from nipype.interfaces.ants import Registration


from narps_open.pipelines import Pipeline
from narps_open.pipelines import TaskInformation

class PipelineTeam08MQ(Pipeline):
    """ A class that defines the pipeline of team 08MQ """

    def __init__(self):
        super().__init__()
        self.fwhm = 6.0
        self.team_id = '08MQ'
        self.contrast_list = []

    def get_preprocessing(self):
        """ Return a Nipype workflow describing the prerpocessing part of the pipeline """

        # IdentityInterface node - allows to iterate over subjects and runs
        info_source = Node(IdentityInterface(), name='info_source')
        info_source.inputs.fields=['subject_id', 'run_id']
        info_source.iterables = [
            ('subject_id', self.subject_list),
            ('run_id', self.run_list),
        ]

        # SelectFiles node - to select necessary files
        file_templates = {
            'anat': join('sub-{subject_id}', 'anat', 'sub-{subject_id}_T1w.nii.gz'),
            'func': join(
                'sub-{subject_id}', 'func', 'sub-{subject_id}_task-MGT_run-{run_id}_bold.nii.gz'
                ),
            'magnitude': join('sub-{subject_id}', 'fmap', 'sub-{subject_id}_magnitude1.nii.gz'),
            'phasediff': join('sub-{subject_id}', 'fmap', 'sub-{subject_id}_phasediff.nii.gz')
        }
        select_files = Node(SelectFiles(file_templates), name = 'select_files')
        select_files.input.base_directory = self.directories.dataset_dir

        # DataSink Node - store the wanted results in the wanted repository
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir

        # FAST Node - Bias field correction on anatomical images
        bias_field_correction = Node(FAST(), name = 'bias_field_correction')
        bias_field_correction.inputs.img_type = 1 # T1 image
        bias_field_correction.inputs.output_biascorrected = True
        #bias_field_correction.inputs.output_biasfield = True

        # BET Node - Brain extraction for anatomical images
        brain_extraction_anat = Node(BET(), name = 'brain_extraction_anat')
        brain_extraction_anat.inputs.frac = 0.5

        # FAST Node - Segmentation of anatomical images
        segmentation_anat = Node(FAST(), name = 'segmentation_anat')
        segmentation_anat.inputs.no_bias = True # Bias field was already removed
        segmentation_anat.inputs.number_classes = 
        segmentation_anat.inputs.segments = True # One image per tissue class

        # ANTs Node - Registration to T1 MNI152 space
        registration_anat = Node(Registration(), name = 'registration_anat')
        registration_anat.inputs.fixed_image = ''
        registration_anat.inputs.moving_image = ''
        registration_anat.inputs.initial_moving_transform = ''
        registration_anat.inputs.transforms = ['Rigid', 'Affine', 'SyN']
        registration_anat.inputs.metric = ['MI', 'MI', 'CC']

        # ErodeImage Node - Erode white-matter mask
        erode_white_matter = Node(ErodeImage(), name = 'erode_white_matter')

        # ErodeImage Node - Erode CSF mask
        erode_csf = Node(ErodeImage(), name = 'erode_csf')

        # BET Node - Brain extraction of magnitude images
        brain_extraction_magnitude = Node(BET(), name = 'brain_extraction_magnitude')
        brain_extraction_magnitude.inputs.frac = 0.5

        # PrepareFieldmap Node - Convert phase and magnitude to fieldmap images
        convert_to_fieldmap = Node(PrepareFieldmap(), name = 'convert_to_fieldmap')

        # BET Node - Brain extraction for functional images
        brain_extraction_func = Node(BET(), name = 'brain_extraction_func')
        brain_extraction_func.inputs.frac = 0.3

        # MCFLIRT Node - Motion correction of functional images
        motion_correction = Node(MCFLIRT(), name = 'motion_correction')
        motion_correction.inputs.cost = 'normcorr'
        motion_correction.inputs.interpolation = 'trilinear'

        # SliceTimer Node - Slice time correction
        slice_time_correction = Node(SliceTimer(), name = 'slice_time_correction')
        slice_time_correction.inputs.time_repetition = TaskInformation()['RepetitionTime']

        custom_order (a pathlike object or string representing an existing file) – Filename of single-column custom interleave order file (first slice is referred to as 1 not 0). Maps to a command-line argument: --ocustom=%s.
        custom_timings (a pathlike object or string representing an existing file) – Slice timings, in fractions of TR, range 0:1 (default is 0.5 = no shift). Maps to a command-line argument: --tcustom=%s.
        environ (a dictionary with keys which are a bytes or None or a value of class ‘str’ and with values which are a bytes or None or a value of class ‘str’) – Environment variables. (Nipype default value: {})
        global_shift (a float) – Shift in fraction of TR, range 0:1 (default is 0.5 = no shift). Maps to a command-line argument: --tglobal.
        index_dir (a boolean) – Slice indexing from top to bottom. Maps to a command-line argument: --down.
        interleaved (a boolean) – Use interleaved acquisition. Maps to a command-line argument: --odd.
        out_file (a pathlike object or string representing a file) – Filename of output timeseries. Maps to a command-line argument: --out=%s.
        output_type (‘NIFTI’ or ‘NIFTI_PAIR’ or ‘NIFTI_GZ’ or ‘NIFTI_PAIR_GZ’) – FSL output type.
        slice_direction (1 or 2 or 3) – Direction of slice acquisition (x=1, y=2, z=3) - default is z. Maps to a command-line argument: --direction=%d.
        time_repetition (a float) – Specify TR of data - default is 3s. Maps to a command-line argument: --repeat=%f.

        # [INFO] The following part has to be modified with nodes of the pipeline
        """
        Anatomical:
            V Bias correction -> Bias field correction was applied to the anatomical images using FAST.
            V Brain extraction -> BET was used for brain extraction for the anatomical, field map, and functional images. A fractional intensity threshold of 0.5 was used for the anatomical and field map images. One of 0.3 was used for the functional data.
            V Segmentation -> Structural images were segmented with FAST. Bias correction was done first.
            Alignment to MNI template ->
                Data were converted to T1 MNI152 space with a 2mm resolution.
                Alignment between T1 anatomical images and the T1 MNI template was calculated with ANTs.
                T1 images had bias field correction applied prior to alignment.
                Rigid (mutual information cost function), affine (mutual information cost function),
                    and SyN (cross correlation cost function) steps were applied, in that order.
                The combined functional-to-anatomical plus distortion correction warps were applied to functional data and then
                    the anatomical-to-MNI warps applied to that data.
            Creation of white matter and CSF masks from segmentation with threshold=1. Erode masks

        Field maps:
            V Brain extraction of magnitude image -> BET was used for brain extraction for the anatomical, field map, and functional images. A fractional intensity threshold of 0.5 was used for the anatomical and field map images. One of 0.3 was used for the functional data.
            V Conversion of phase and magnitude images to field maps

        High contrast functional volume:
            Alignment to anatomical image including distortion correction with field map
            Calculation of inverse warp (anatomical to functional)

        Functional:
            V Brain extraction -> BET was used for brain extraction for the anatomical, field map, and functional images. A fractional intensity threshold of 0.5 was used for the anatomical and field map images. One of 0.3 was used for the functional data.
            V Motion correction with high contrast image as reference -> MCFLIRT was used for motion correction.
                The single volume, high contrast image was used as the reference scan.
                Normalised correlation was used as the image similarity metric with trilinear interpolation.
            Slice time correction -> Slicetimer was used and was applied after motion correction.
                The middle slice was used as the reference slice. Sinc interpolation was used.
            Alignment of white matter and CSF masks to functional space with previously calculated warps
            Calculate aCompCor components
        """

        preprocessing = Workflow(base_dir = self.directories.working_dir, name = 'preprocessing')
        preprocessing.connect([
            # Inputs
            (info_source, select_files, [('subject_id', 'subject_id'), ('run_id', 'run_id')]),
            (select_files, node_name, [('func', 'node_input_name')]),
            (node_name, data_sink, [('node_output_name', 'preprocessing.@sym_link')]),

            # Anatomical images
            (select_files, bias_field_correction, [('anat', 'in_files')]),
            (bias_field_correction, brain_extraction_anat, [('restored_image', 'in_file')]),
            (brain_extraction_anat, segmentation_anat, [('out_file', 'in_file')]),
            (segmentation_anat, registration_anat, [('?', 'in_file')]),

            (registration_anat, erode_white_matter, [('', '')]),
            (registration_anat, erode_csf, [('', '')]),

            (erode_white_matter, , [('', '')]),
            (erode_csf, , [('', '')]),

            # Field maps
            (select_files, brain_extraction_magnitude, [('magnitude', 'in_file')]),
            (brain_extraction_magnitude, convert_to_fieldmap, [('out_file', 'in_magnitude')]),
            (select_files, convert_to_fieldmap, [('phasediff', 'in_phase')]),

            # High contrast functional volume
            # Functional images
            (select_files, brain_extraction_func, [('func', 'in_file')]),
            (brain_extraction_func, motion_correction, [('out_file', 'in_file')]),
            (, motion_correction, [('out_file', 'ref_file')]), # high contrast images
            (motion_correction, slice_time_correction, [('out_file', 'in_file')]),

        ])

        return preprocessing

    def get_run_level_analysis(self):
        """ Return a Nipype workflow describing the run level analysis part of the pipeline """
        return None

    def get_session_infos(event_file: str):
        """
        Create Bunchs for specifyModel.

        Parameters :
        - event_file : file corresponding to the run and the subject to analyze

        Returns :
        - subject_info : list of Bunch for 1st level analysis.
        """

        """
        Canonical double gamma HRF plus temporal derivative.
        Model consisted of:

        Event regressor with 4 second ON duration.
        Parametric modulation of events corresponding to gain magnitude. Mean centred.
        Parametric modulation of events corresponding to loss magnitude. Mean centred.
        Response regressor with 1 for accept and -1 for reject. Mean centred.
        Six head motion parameters plus four aCompCor regressors.
        Model and data had a 90s high-pass filter applied.
        """

        from nipype.interfaces.base import Bunch

        condition_names = ['trial', 'gain', 'loss']

        onset = {}
        duration = {}
        amplitude = {}

        # Creates dictionary items with empty lists for each condition.
        for condition in condition_names:  
            onset.update({condition: []}) 
            duration.update({condition: []})
            amplitude.update({condition: []})

        with open(event_file, 'rt') as file:
            next(file)  # skip the header

            for line in file:
                info = line.strip().split()
                # Creates list with onsets, duration and loss/gain for amplitude (FSL)
                for condition in condition_names:
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

        subject_info = []
        subject_info.append(
            Bunch(
                conditions = condition_names,
                onsets = [onset[k] for k in condition_names],
                durations = [duration[k] for k in condition_names],
                amplitudes = [amplitude[k] for k in condition_names],
                regressor_names = None,
                regressors = None,
            )
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
        effect_loss = ('effect_of_loss', 'T', conditions, [0, 0, 1])

        # Contrast list
        return [trial, effect_gain, effect_loss]

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
                'preprocess',
                '_run_id_*_subject_id_{subject_id}',
                'complete_filename_{subject_id}_complete_filename.nii',
            ),
            'event': join(
                self.directories.dataset_dir,
                'sub-{subject_id}',
                'func',
                'sub-{subject_id}_task-MGT_run-*_events.tsv',
            )
        }

        # SelectFiles node - to select necessary files
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir

        # DataSink Node - store the wanted results in the wanted repository
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir

        # [INFO] This is the node executing the get_subject_infos_spm function
        # Subject Infos node - get subject specific condition information
        subject_infos = Node(
            Function(
                function = self.get_subject_infos,
                input_names = ['event_files', 'runs'],
                output_names = ['subject_info']                
            ),
            name = 'subject_infos',
        )
        subject_infos.inputs.runs = self.run_list

        # [INFO] This is the node executing the get_contrasts function
        # Contrasts node - to get contrasts
        contrasts = Node(
            Function(
                function = self.get_contrasts,
                input_names = ['subject_id'],
                output_names = ['contrasts']                
            ),
            name = 'contrasts',
        )

        # [INFO] The following part has to be modified with nodes of the pipeline

        # [TODO] For each node, replace 'node_name' by an explicit name, and use it for both:
        #   - the name of the variable in which you store the Node object
        #   - the 'name' attribute of the Node
        # [TODO] The node_function refers to a NiPype interface that you must import
        # at the beginning of the file.
        node_name = Node(
            node_function,
            name = 'node_name'
        )

        # [TODO] Add other nodes with the different steps of the pipeline

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
                info_source,
                contrasts,
                [('subject_id', 'subject_id')]
            ),
            (
                select_files,
                subject_infos,
                [('event', 'event_files')]
            ),
            (
                select_files,
                node_name,
                [('func', 'node_input_name')]
            ),
            (
                node_name, data_sink,
                [('node_output_name', 'preprocess.@sym_link')]
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

        return copes_equal_indifference, copes_equal_range,
            varcopes_equal_indifference, varcopes_equal_range,
            equal_indifference_id, equal_range_id,
            copes_global, varcopes_global


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
        elif method == 'equalIndifference':
            regressors = dict(group_mean = [1 for i in range(len(equal_indifference_id))])

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
                output_names = ['regressors'],
                function = self.get_regressors,
            ),
            name = 'regs',
        )
        regs.inputs.method = method
        regs.inputs.subject_list = subject_list

        # [INFO] The following part has to be modified with nodes of the pipeline

        # [TODO] For each node, replace 'node_name' by an explicit name, and use it for both:
        #   - the name of the variable in which you store the Node object
        #   - the 'name' attribute of the Node
        # [TODO] The node_function refers to a NiPype interface that you must import
        # at the beginning of the file.
        node_name = Node(
            node_function,
            name = 'node_name'
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
                    select_files,
                    node_name[('func', 'node_input_name')],
                ),
                (
                    node_variable,
                    datasink_groupanalysis,
                    [('node_output_name', 'preprocess.@sym_link')],
                ),
            ]
        ) # Complete with other links between nodes

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
