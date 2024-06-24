#!/usr/bin/python
# coding: utf-8

""" Write the work of NARPS team L3V8 using Nipype """

from os.path import join
from itertools import product

from nipype import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Function, Rename, Merge
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.algorithms.misc import Gunzip
from nipype.interfaces.spm import (Realign, Coregister, NewSegment,
    Normalize12, Smooth, OneSampleTTestDesign, EstimateModel, 
    EstimateContrast, Level1Design, TwoSampleTTestDesign, Threshold)
from nipype.interfaces.spm.base import Info as SPMInfo
from nipype.interfaces.fsl import ExtractROI
from nipype.algorithms.modelgen import SpecifySPMModel

from narps_open.pipelines import Pipeline
from narps_open.data.task import TaskInformation
from narps_open.data.participants import get_group
from narps_open.core.common import (
    remove_parent_directory, list_intersection, elements_in_string, clean_list
    )
from narps_open.utils.configuration import Configuration

class L3V8(Pipeline):
    """ NARPS team L3V8 pipeline class definition and description of pipeline steps 
    Participants:101"""
    
    def __init__(self):
        super().__init__()
        self.fwhm = 6.0
        self.team_id = 'L3V8'
        self.contrast_list = ['0001', '0002', '0003', '0004', '0005', '0006']

        # Define contrasts
        gain_conditions = [f'gamble_run{r}xgain_run{r}^1' for r in range(1,len(self.run_list) + 1)]
        loss_conditions = [f'gamble_run{r}xloss_run{r}^1' for r in range(1,len(self.run_list) + 1)]
        self.subject_level_contrasts = [
            ('pos_gain', 'T', gain_conditions, [1, 1, 1, 1]), # average postivite activation -gain value
            ('pos_gain', 'T', gain_conditions, [1, 1, 1, 1]), # average postivite activation -gain value
            ('pos_loss', 'T', loss_conditions, [1, 1, 1, 1]), # average positive effect -loss value
            ('neg_gain', 'T', gain_conditions, [-1, -1, -1, -1]),  # average negative activation - gain value
            ('neg_loss', 'T', loss_conditions, [-1, -1, -1, -1]) # average negative activation - loss value
            # figure out group contrast settings # TODO
            ('contrast1', 'T', gain_conditions, [-1, 1, -1, 1, -1, 1, -1, 1]) # group differences between EqI and EqR
            
            ]
    def get_preprocessing_sub_workflow(self):
        """
        Create the second part of the preprocessing workflow.

        Returns:
            - preprocessing : nipype.WorkFlow
        """
        # Infosource Node - To iterate on subjects
        information_source = Node(IdentityInterface(
            fields = ['subject_id', 'run_id']),
            name = 'information_source')
        information_source.iterables = [
            ('subject_id', self.subject_list), ('run_id', self.run_list)
            ]

        # SelectFiles Node - Select necessary files
        templates = {
            'anat' : join('sub-{subject_id}', 'anat', 'sub-{subject_id}_T1w.nii.gz'),
            'func' : join('sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold.nii.gz'),
        }
        # select files
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir
        # Realignment, co-registration, segmentation, normalization, smooth (6 mm).
        # Gunzip nodes - gunzip files because SPM do not use .nii.gz files
        gunzip_anat = Node(Gunzip(), name = 'gunzip_anat')
        gunzip_func = Node(Gunzip(), name = 'gunzip_func')            

        # motion correction - default realignment node in SPM

        motion_correction = Node(interface=Realign(), name='realign')
        motion_correction.inputs.register_to_mean = False # they registered to 1st slice
        motion_correction.inputs.fwhm = 6
        motion_correction.inputs.interp = 4
        motion_correction.inputs.quality = 0.9
        motion_correction.inputs.separation = 4
        motion_correction.inputs.wrap = [0, 0, 0]
        motion_correction.inputs.write_which = [2, 1]
        motion_correction.inputs.write_interp = 4 
        motion_correction.inputs.write_wrap = [0, 0, 0]
        motion_correction.inputs.write_mask = True
        motion_correction.inputs.jobtype = 'estwrite'


        # coregistration node

        coregister = Node(Coregister(), name="coregister")
        coregister.inputs.jobtype = 'estimate'
        coregister.inputs.cost_function = 'nmi'
        coregister.inputs.fwhm = [6.0, 6.0]
        coregister.inputs.separation = [4.0, 2.0]
        coregister.inputs.tolerance = [0.02, 0.02, 0.02, 0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001]

        # Get SPM Tissue Probability Maps file
        spm_tissues_file = join(SPMInfo.getinfo()['path'], 'tpm', 'TPM.nii')

        # Segmentation Node - SPM Segment function via custom scripts (defaults left in place)
        segmentation = Node(NewSegment(), name = 'segmentation')
        segmentation.inputs.write_deformation_fields = [True, True]
        segmentation.inputs.channel_info = (0.0001, 60, (True, True))
        segmentation.inputs.affine_regularization = 'mni'
        segmentation.inputs.warping_regularization = [0, 0.001, 0.5, 0.05, 0.2]
        segmentation.inputs.sampling_distance = 3
        segmentation.inputs.tissues = [
            [(spm_tissues_file, 1), 1, (True,False), (True, False)],
            [(spm_tissues_file, 2), 1, (True,False), (True, False)],
            [(spm_tissues_file, 3), 2, (True,False), (False, False)],
            [(spm_tissues_file, 4), 3, (True,False), (False, False)],
            [(spm_tissues_file, 5), 4, (True,False), (False, False)],
            [(spm_tissues_file, 6), 2, (False,False), (False, False)]
        ]

        # normalization node

        normalize = Node(Normalize12(), name="normalize") #old normalize now
        normalize.inputs.jobtype = 'write'
        normalize.inputs.write_voxel_sizes = [3, 3, 3]
        normalize.inputs.write_interp = 4
        normalize.inputs.warping_regularization = [0, 0.001, 0.5, 0.05, 0.2]
        # smoothing node

        smooth = Node(Smooth(), name="smooth")
        smooth.inputs.fwhm = [6, 6, 6]
        smooth.inputs.implicit_masking = False

        # DataSink Node - store the wanted results in the wanted repository
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir

        # Create preprocessing workflow and connect its nodes
        preprocessing =  Workflow(base_dir = self.directories.working_dir, name = 'preprocessing')
        preprocessing.config['execution']['stop_on_first_crash'] = 'true'
        preprocessing.connect([
            (information_source, select_files, [
                ('subject_id', 'subject_id'),
                ('run_id', 'run_id')]),
            (select_files, gunzip_anat, [('anat', 'in_file')]),
            (select_files, gunzip_func, [('func', 'in_file')]),
            (gunzip_func, motion_correction, [('out_file', 'in_files')]),
            (motion_correction, data_sink, [('realigned_files', 'preprocessing.@realigned_files'),
                                           ('realignment_parameters', 'preprocessing.@realignment_parameters'),
                                           ('mean_image', 'preprocessing.@mean_image')]),
            (motion_correction, coregister, [('mean_image', 'source')]),                               
            (gunzip_anat, coregister, [('out_file', 'source')]),
            (coregister, segmentation, [('coregistered_files', 'channel_files')]),
            (gunzip_anat, segmentation, [('out_file', 'channel_files')]),
            (segmentation, data_sink, [('bias_corrected_images', 'preprocessing.@bias_corrected_images'),
                                       ('forward_deformation_field', 'preprocessing.@forward_deformation_field'),
                                       ('backward_deformation_field', 'preprocessing.@backward_deformation_field')]),
            (segmentation, normalize, [('forward_deformation_field', 'deformation_file')]),
            (normalize, smooth, [('normalized_files', 'in_files')]), # check normalization anat
            (smooth, data_sink, [('smoothed_files', 'preprocessing.@smoothed_files')]),
        ])    

        # Remove large files, if requested
        if Configuration()['pipelines']['remove_unused_data']:

            # Merge Node - Merge file names to be removed after datasink node is performed
            merge_removable_files = Node(Merge(5), name = 'merge_removable_files')
            merge_removable_files.inputs.ravel_inputs = True

            # Function Nodes remove_files - Remove sizeable files once they aren't needed
            remove_after_datasink = MapNode(Function(
                function = remove_parent_directory,
                input_names = ['_', 'file_name'],
                output_names = []
                ), name = 'remove_after_datasink', iterfield = 'file_name')

            # Add connections
            preprocessing.connect([
                (gunzip_func, merge_removable_files, [('out_file', 'in1')]),
                (gunzip_anat, merge_removable_files, [('out_file', 'in2')]),
                (select_files, merge_removable_files, [('out_file', 'in3')]),
                (coregister, merge_removable_files, [('coregistered_files', 'in4')]),
                (segmentation, merge_removable_files, [('native_class_images', 'in5')]),
                (merge_removable_files, remove_after_datasink, [('out', 'file_name')]),
                (data_sink, remove_after_datasink, [('out_file', '_')])
            ])
        return preprocessing
    
    def get_preprocessing(self):
        """
        Create the full preprocessing workflow.

        Returns: a list of nipype.WorkFlow
        """
        return [
            self.get_preprocessing_sub_workflow()
        ]
    
    # TODO edit below code.
    def get_preprocessing_outputs(self):
        """ Return the names of the files the preprocessing is supposed to generate. """

        # Outputs from dartel workflow
        return_list = [join(self.directories.output_dir, 'dartel_template', 'template_6.nii')]
        return_list += [join(self.directories.output_dir, 'dartel_template',
            f'u_rc1subject_id_{subject_id}_struct_template.nii')\
            for subject_id in self.subject_list]

        # Outputs from preprocessing
        parameters = {
            'subject_id': self.subject_list,
            'run_id': self.run_list,
        }
        parameter_sets = product(*parameters.values())
        output_dir = join(
            self.directories.output_dir,
            'preprocessing',
            '_run_id_{run_id}_subject_id_{subject_id}'
        )
        templates = [
            # Realignment parameters
            join(output_dir, 'rp_asub-{subject_id}_task-MGT_run-{run_id}_bold.txt'),
            # Realigned unwarped files
            join(output_dir, 'asub-{subject_id}_task-MGT_run-{run_id}_bold.nii'),
            # Normalized_files
            join(output_dir, 'swasub-{subject_id}_task-MGT_run-{run_id}_bold.nii'),
            # Normalized class images
            join(output_dir, 'wc2sub-{subject_id}_T1w.nii'),
            join(output_dir, 'wc1sub-{subject_id}_T1w.nii')
        ]
        return_list += [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets for template in templates]

        return return_list

    