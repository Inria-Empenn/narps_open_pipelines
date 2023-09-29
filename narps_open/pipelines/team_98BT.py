from nipype.interfaces.spm import (Coregister, Smooth, OneSampleTTestDesign, EstimateModel, EstimateContrast, 
                                   Level1Design, TwoSampleTTestDesign, RealignUnwarp, NewSegment, SliceTiming,
                                  DARTEL, DARTELNorm2MNI, FieldMap)
from nipype.interfaces.spm import Threshold
from nipype.interfaces.fsl import ApplyMask, ExtractROI

import niflow.nipype1.workflows.fmri.spm as spm_wf  # spm
from nipype.algorithms.modelgen import SpecifySPMModel
from nipype.interfaces.utility import IdentityInterface, Function, Rename
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.algorithms.misc import Gunzip
from nipype import Workflow, Node, MapNode, JoinNode
from nipype.interfaces.base import Bunch

from os.path import join as opj
import os
import json

def get_dartel_template_wf(exp_dir, result_dir, working_dir, output_dir, subject_list):
    

    infosource_dartel = Node(IdentityInterface(fields=['subject_id']), name="infosource_dartel")
    infosource_dartel.iterables = ('subject_id', subject_list)
    
    # Templates to select files node
    anat_file = opj('sub-{subject_id}', 'anat', 
                    'sub-{subject_id}_T1w.nii.gz')

    template = {'anat' : anat_file}

    # SelectFiles node - to select necessary files
    selectfiles_dartel = Node(SelectFiles(template, base_directory=exp_dir), name = 'selectfiles_dartel')
    
    # GUNZIP NODE : SPM do not use .nii.gz files
    gunzip_anat = Node(Gunzip(), name = 'gunzip_anat')
    
    def get_dartel_input(structural_files):
        print(structural_files)
        return structural_files

    dartel_input = JoinNode(Function(input_names = ['structural_files'],
                                output_names = ['structural_files'],
                                function = get_dartel_input), name = 'dartel_input', 
                            joinsource="infosource_dartel", joinfield="structural_files")
    
    rename_dartel = MapNode(Rename(format_string="subject_id_%(subject_id)s_struct"),
                            iterfield=['in_file', 'subject_id'],
                            name='rename_dartel')
    
    rename_dartel.inputs.subject_id = subject_list
    rename_dartel.inputs.keep_ext = True

    dartel_workflow = spm_wf.create_DARTEL_template(name='dartel_workflow')
    dartel_workflow.inputs.inputspec.template_prefix = "template"
    
    # DataSink Node - store the wanted results in the wanted repository
    datasink_dartel = Node(DataSink(base_directory=result_dir, container=output_dir), 
                            name='datasink_dartel')
    
    dartel =  Workflow(base_dir = opj(result_dir, working_dir), name = "dartel")

    dartel.connect([(infosource_dartel, selectfiles_dartel, [('subject_id', 'subject_id')]),
                    (selectfiles_dartel, gunzip_anat, [('anat', 'in_file')]),
                    (gunzip_anat, dartel_input, [('out_file', 'structural_files')]),
                    (dartel_input, rename_dartel, [('structural_files', 'in_file')]),
                    (rename_dartel, dartel_workflow, [('out_file', 'inputspec.structural_files')]),
                    (dartel_workflow, datasink_dartel, 
                     [('outputspec.template_file', 'dartel_template.@template_file'),
                     ('outputspec.flow_fields', 'dartel_template.@flow_fields')])])
    
    return dartel


def get_fieldmap_infos(info_fmap, magnitude):
    """
    Function to get information necessary to compute the fieldmap. 

    Parameters: 
        - info_fmap: str, file with fieldmap information
        - magnitude: list of str, list of magnitude files

    Returns: 
        - TE: float, echo time obtained from fieldmap information file
        - magnitude_file: str, necessary file to compute fieldmap
    """
    import json 
    
    with open(info_fmap, 'rt') as fp:
        fmap_info = json.load(fp)
        
    short_TE = min(float(fmap_info['EchoTime1']), float(fmap_info['EchoTime2']))
    long_TE = max(float(fmap_info['EchoTime1']), float(fmap_info['EchoTime2']))
    if short_TE == float(fmap_info['EchoTime1']):
        magnitude_file = magnitude[0]
    elif short_TE == float(fmap_info['EchoTime2']):
        magnitude_file = magnitude[1]
        
    TE = (short_TE, long_TE)
        
    return TE, magnitude_file

def rm_field_files(files, subject_id, run_id, result_dir, working_dir):
    import shutil
    from os.path import join as opj

    preproc_dir = opj(result_dir, working_dir, 'preprocessing', f"run_id_{run_id}_subject_id_{subject_id}")
    
    dir_to_rm = ['gunzip_func', 'gunzip_phasediff', 'fieldmap_infos', 'gunzip_magnitude', 'fieldmap', 
                'slice_timing']
    for dirs in dir_to_rm:
        try:
            shutil.rmtree(opj(preproc_dir, dirs))
        except OSError as e:
            print(e)
        else:
            print(f"The directory {dirs} is deleted successfully")
    
    return files

def get_preprocessing(exp_dir, result_dir, working_dir, output_dir, subject_list, run_list, fwhm, N, ST, TA, TR, total_readout_time): 
    
    infosource = Node(IdentityInterface(fields=['subject_id', 'run_id']), name="infosource")

    infosource.iterables = [('subject_id', subject_list), ('run_id', run_list)]

    # Templates to select files node
    anat_file = opj('sub-{subject_id}', 'anat', 
                    'sub-{subject_id}_T1w.nii.gz')

    func_file = opj('sub-{subject_id}', 'func', 
                    'sub-{subject_id}_task-MGT_run-{run_id}_bold.nii.gz')

    magnitude_file = opj('sub-{subject_id}', 'fmap', 'sub-{subject_id}_magnitude*.nii.gz')

    phasediff_file = opj('sub-{subject_id}', 'fmap', 'sub-{subject_id}_phasediff.nii.gz')

    info_fieldmap_file = opj('sub-{subject_id}', 'fmap', 'sub-{subject_id}_phasediff.json')
    
    dartel_flowfields_file = opj(result_dir, output_dir, 'dartel_template', 
                                 'u_rc1subject_id_{subject_id}_struct_template.nii')
    
    dartel_template_file = opj(result_dir, output_dir, 'dartel_template', 'template_6.nii')

    template = {'anat' : anat_file, 'func' : func_file, 'magnitude' : magnitude_file, 'phasediff' : phasediff_file,
                'info_fmap' : info_fieldmap_file, 'dartel_template' : dartel_template_file, 
                'dartel_flow_field' : dartel_flowfields_file}

    # SelectFiles node - to select necessary files
    selectfiles_preproc = Node(SelectFiles(template, base_directory=exp_dir), name = 'selectfiles_preproc')

    # GUNZIP NODE : SPM do not use .nii.gz files
    gunzip_anat = Node(Gunzip(), name = 'gunzip_anat')

    gunzip_func = Node(Gunzip(), name = 'gunzip_func')

    gunzip_magnitude = Node(Gunzip(), name = 'gunzip_magnitude')

    gunzip_phasediff = Node(Gunzip(), name = 'gunzip_phasediff')


    fieldmap_infos = Node(Function(input_names = ['info_fmap', 'magnitude'],
                                      output_names = ['TE', 'magnitude_file'],
                                      function = get_fieldmap_infos), name = 'fieldmap_infos')

    fieldmap = Node(FieldMap(blip_direction = -1), name = 'fieldmap')

    fieldmap.inputs.total_readout_time = total_readout_time

    """
    - **Segmentation** : SPM Segment function via custom scripts (defaults left in place)
    """
    
    tissue1 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 1), 1, (True,False), (True, False)]
    tissue2 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 2), 1, (True,False), (True, False)]
    tissue3 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 3), 2, (True,False), (False, False)]
    tissue4 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 4), 3, (True,False), (False, False)]
    tissue5 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 5), 4, (True,False), (False, False)]
    tissue6 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 6), 2, (False,False), (False, False)]
    
    tissue_list = [tissue1, tissue2, tissue3, tissue4, tissue5, tissue6]
    
    segmentation = Node(NewSegment(write_deformation_fields = [True, True], tissues = tissue_list, 
                                  channel_info = (0.0001, 60, (True, True))), name = "segmentation")

    """
    - **Slice timing** : SPM slice time correction with default parameters 
    (shift via sinc interpolation and Fourier transform), with ref =2. Slice time correction performed 
    as first step of preprocessing, prior to motion correction.
    """

    slice_timing = Node(SliceTiming(num_slices = N, ref_slice = 2, slice_order = ST, 
                                   time_acquisition = TA, time_repetition = TR), name = "slice_timing")

    #in_files = func
    """
    - **Motion correction** : SPM realign and unwarp, incorporating field maps into motion correction for unwarping.
    Most parameters as SPM default, but 4th degree bslines interpolation was used for estimation. 
    Images were rereferenced to the first scan in each run. 
    """

    motion_correction = Node(RealignUnwarp(interp = 4), name = "motion_correction")

    #in_files = timecorrected_files

    """
    - **Intrasubject coregistration** : Each EPI scan was coregistered to the structural scan via SPM's coreg 
    function, using normalized mutual information. 
    """
    extract_first = Node(ExtractROI(t_min = 1, t_size=1, output_type = 'NIFTI'), name = 'extract_first')
    
    coregistration = Node(Coregister(cost_function = 'nmi', jobtype = 'estimate'), name = "coregistration")

    #target = anat
    #source = realigned_unwarped_files

    dartel_norm_func = Node(DARTELNorm2MNI(fwhm = fwhm, modulate = False, voxel_size = (2.3, 2.3, 2.15)), name = "dartel_norm_func")
    #apply_to_files = coregistered_source
    
    dartel_norm_anat = Node(DARTELNorm2MNI(fwhm = fwhm, voxel_size = (1, 1, 1)), name = "dartel_norm_anat")

    remove_field_files = Node(Function(input_names = ['files', 'subject_id', 'run_id', 'result_dir', 'working_dir'],
                                    output_names = ['files'],
                                    function = rm_field_files), name = 'remove_field_files')

    remove_field_files.inputs.result_dir = result_dir
    remove_field_files.inputs.working_dir = working_dir

    # DataSink Node - store the wanted results in the wanted repository
    datasink_preproc = Node(DataSink(base_directory=result_dir, container=output_dir), name='datasink_preproc')
    
    #dartel = get_dartel_template_wf(exp_dir, result_dir, working_dir, output_dir, subject_list)

    preprocessing =  Workflow(base_dir = opj(result_dir, working_dir), name = "preprocessing")

    preprocessing.connect([(infosource, selectfiles_preproc, [('subject_id', 'subject_id'), 
                                                             ('run_id', 'run_id')]),
                           (infosource, remove_field_files, [('subject_id', 'subject_id'), 
                                                            ('run_id', 'run_id')]),
                           (selectfiles_preproc, gunzip_anat, [('anat', 'in_file')]),
                           (selectfiles_preproc, gunzip_func, [('func', 'in_file')]),
                           (selectfiles_preproc, gunzip_phasediff, [('phasediff', 'in_file')]),
                           (selectfiles_preproc, fieldmap_infos, [('info_fmap', 'info_fmap'), 
                                                                  ('magnitude', 'magnitude')]),
                           (fieldmap_infos, gunzip_magnitude, [('magnitude_file', 'in_file')]),
                           (fieldmap_infos, fieldmap, [('TE', 'echo_times')]),
                           (gunzip_magnitude, fieldmap, [('out_file', 'magnitude_file')]),
                           (gunzip_phasediff, fieldmap, [('out_file', 'phase_file')]),
                           (gunzip_func, fieldmap, [('out_file', 'epi_file')]),
                           (fieldmap, motion_correction, [('vdm', 'phase_map')]),
                           (gunzip_anat, segmentation, [('out_file', 'channel_files')]),
                           (gunzip_func, slice_timing, [('out_file', 'in_files')]),
                           (slice_timing, motion_correction, [('timecorrected_files', 'in_files')]),
                           (motion_correction, remove_field_files, [('realigned_unwarped_files', 'files')]), 
                           (remove_field_files, coregistration, [('files', 'apply_to_files')]),
                           (gunzip_anat, coregistration, [('out_file', 'target')]),
                           (remove_field_files, extract_first, [('files', 'in_file')]),
                           (extract_first, coregistration, [('roi_file', 'source')]),
                           (selectfiles_preproc, dartel_norm_func, [('dartel_flow_field', 'flowfield_files'), 
                                                                   ('dartel_template', 'template_file')]),
                           (selectfiles_preproc, dartel_norm_anat, [('dartel_flow_field', 'flowfield_files'), 
                                                                  ('dartel_template', 'template_file')]),
                           (gunzip_anat, dartel_norm_anat, [('out_file', 'apply_to_files')]),
                           (coregistration, dartel_norm_func, [('coregistered_files', 'apply_to_files')]),
                           (dartel_norm_func, datasink_preproc, 
                            [('normalized_files', 'preprocessing.@normalized_files')]),
                           #(dartel_norm_anat, datasink_preproc, 
                           # [('normalized_files', 'preprocessing.@normalized_anat')]),
                           (motion_correction, datasink_preproc, 
                            [('realigned_unwarped_files', 'preprocessing.@motion_corrected'), 
                             ('realignment_parameters', 'preprocessing.@param')]), 
                           (segmentation, datasink_preproc, [('normalized_class_images', 'preprocessing.@seg')])
                          ])
    
    return preprocessing

def compute_parameters(parameters_files, wc2_file, motion_corrected_files, subject_id, result_dir, working_dir):
    import pandas as pd
    from os.path import join as opj 
    import os
    import nibabel as nib
    import numpy as np
    from nilearn import image
    from nilearn.image import resample_to_img, resample_img
    # import warnings filter
    from warnings import simplefilter
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=RuntimeWarning)
    
    mean_wm = [[] for i in range(len(motion_corrected_files))]
    
    wc2 = nib.load(wc2_file)
    wc2_mask = wc2.get_fdata() > 0.6
    wc2_mask = wc2_mask.astype(int)
    

    for i, functional_file in enumerate(sorted(motion_corrected_files)):
        functional = nib.load(functional_file)
        
        
        for slices in image.iter_img(functional):
            slice_img = resample_to_img(slices, wc2, interpolation='nearest', clip = True)

            slice_data = slice_img.get_fdata()
            masked_slice = slice_data * wc2_mask
            mean_wm[i].append(np.mean(masked_slice))
    
    new_parameters_files = []
    for i, file in enumerate(sorted(parameters_files)):
        df = pd.read_table(file, sep = '  ', header = None)
        
        df['Mean_WM'] = mean_wm[i]
        
        new_path =opj(result_dir, working_dir, 'parameters_file', 
                      f"parameters_file_sub-{subject_id}_run{'0' + str(i+1)}.tsv")
        if not os.path.isdir(opj(result_dir, working_dir)):
            os.mkdir(opj(result_dir, working_dir))
        if not os.path.isdir(opj(result_dir, working_dir, 'parameters_file')):
            os.mkdir(opj(result_dir, working_dir, 'parameters_file'))
        writer = open(new_path, "w")
        writer.write(df.to_csv(sep = '\t', index = False, header = False, na_rep = '0.0'))
        writer.close()

        new_parameters_files.append(new_path)

    return new_parameters_files

"""
An event related design was used. Events were entered as a single events with a 4 second duration. 
Paramteric modualtors were added for each condition for gain, loss, and for accept or reject 
(in that order; note that SPM performs serial orthogonalization of modulators).

Canonical HRFs were used along with the derivative and dispersion functions.
SPM defaults were used for other analysis parameters (Volterra=1, 1st order autocorrelation, high pass filter=128).
The six motion parameters, the absolute value of the 1st derivative of the six parameters, 
and the mean WM signal were entered as covariates into the model.
Mean WM signal was calculated for each individual by taking their SPM segmentation WM mask, 
thresholded at 0.6, as a mask to the motion corrected BOLD data, taking the mean timeseries of 
all voxels determined to be within the white matter.

All four scans were entered into a single fixed effects model for each individual, as separate sessions. 
For each individual, a contrast map was calculated for the parametric effects of gains and the parametric
effects of losses, with both positive and negative contrasts run, collapsing across runs.
Contrast beta maps for the parametric effects of losses or gains were entered into the second level analysis. 
Mass univariate modelling was used. A group analysis (radom effects) model was run for the equaldifference group, 
equalrange group, and the difference between groups (as separate GLMs)
The main effects of the parametric modulators for gains and losses were tested. The whole brain was used for 
calculating cluster statistics.
The data was thresholded such that only clusters above p<0.05 FDR corrected were kept, and data was then masked 
to answer the specific hypotheses. Cluster extent, using SPM's FDR corrected cluster extent statistic, 
and a threshold of p<0.001 uncorrected.
"""

def get_subject_infos(event_files, runs):
    '''
    Create Bunchs for specifySPMModel.

    Parameters :
    - event_files: list of str, list of events files (one per run) for the subject
    - runs: list of str, list of runs to use
    
    Returns :
    - subject_info : list of Bunch for 1st level analysis.
    '''

    from nipype.interfaces.base import Bunch
    
    cond_names = ['gamble']
    onset = {}
    duration = {}
    weights_gain = {}
    weights_loss = {}
    answers = {}
    
    for r in range(len(runs)):  # Loop over number of runs.
        onset.update({s + '_run' + str(r+1) : [] for s in cond_names}) # creates dictionary items with empty lists
        duration.update({s + '_run' + str(r+1) : [] for s in cond_names}) 
        weights_gain.update({'gain_run' + str(r+1) : []})
        weights_loss.update({'loss_run' + str(r+1) : []})
        answers.update({'answers_run' + str(r+1) : []})
    
    for r, run in enumerate(runs):
        
        f_events = event_files[r]
        
        with open(f_events, 'rt') as f:
            next(f)  # skip the header
            
            for line in f:
                info = line.strip().split()
                
                for cond in cond_names:
                    val = cond + '_run' + str(r+1) # trial_run1
                    val_gain = 'gain_run' + str(r+1) # gain_run1
                    val_loss = 'loss_run' + str(r+1) # loss_run1
                    val_answer = 'answers_run' + str(r+1)
                    onset[val].append(float(info[0])) # onsets for trial_run1 
                    duration[val].append(float(4))
                    weights_gain[val_gain].append(float(info[2])) # weights gain for trial_run1
                    weights_loss[val_loss].append(float(info[3])) # weights loss for trial_run1
                    if "accept" in str(info[5]):
                        answers[val_answer].append(1)
                    else: 
                        answers[val_answer].append(0)

    # Bunching is done per run, i.e. trial_run1, trial_run2, etc.
    # But names must not have '_run1' etc because we concatenate runs 
    subject_info = []
    for r in range(len(runs)):

        cond = [c + '_run' + str(r+1) for c in cond_names]
        gain = 'gain_run' + str(r+1)
        loss = 'loss_run' + str(r+1)
        answer = 'answers_run' + str(r+1)

        subject_info.insert(r,
                           Bunch(conditions=cond,
                                 onsets=[onset[k] for k in cond],
                                 durations=[duration[k] for k in cond],
                                 amplitudes=None,
                                 tmod=None,
                                 pmod=[Bunch(name=[gain, loss, answer],
                                             poly=[1, 1, 1],
                                             param=[weights_gain[gain],
                                                    weights_loss[loss], 
                                                   answers[answer]])],
                                 regressor_names=None,
                                 regressors=None))

    return subject_info

def get_contrasts(subject_id):
    '''
    Create the list of tuples that represents contrasts. 
    Each contrast is in the form : 
    (Name,Stat,[list of condition names],[weights on those conditions])

    Parameters:
        - subject_id: str, ID of the subject 

    Returns:
        - contrasts: list of tuples, list of contrasts to analyze
    '''
    runs = 4

    gamble = []
    gain = []
    loss = []
    for ir in range(runs):
        ir += 1
        gamble.append('gamble_run%i' % ir)
        gain.append('gamble_run%ixgain_run%i^1' % (ir, ir))
        loss.append('gamble_run%ixloss_run%i^1' % (ir, ir))

    pos_1 = [1] * runs
    neg_1 = [-1] * runs


    pos_gain = (
        'pos_gain', 'T',
        gain, pos_1)

    pos_loss = (
        'pos_loss', 'T',
        loss, pos_1)
    
    neg_gain = (
        'neg_gain', 'T',
        gain, neg_1)

    neg_loss = (
        'neg_loss', 'T',
        loss, neg_1)

    contrasts = [pos_gain, pos_loss, neg_gain, neg_loss]

    return contrasts

def get_l1_analysis(subject_list, TR, run_list, exp_dir, result_dir, working_dir, output_dir):
    """
    Returns the first level analysis workflow.

    Parameters: 
        - exp_dir: str, directory where raw data are stored
        - result_dir: str, directory where results will be stored
        - working_dir: str, name of the sub-directory for intermediate results
        - output_dir: str, name of the sub-directory for final results
        - subject_list: list of str, list of subject for which you want to do the analysis
        - run_list: list of str, list of runs for which you want to do the analysis 
        - TR: float, time repetition used during acquisition

    Returns: 
        - l1_analysis : Nipype WorkFlow 
    """
    # Infosource Node - To iterate on subjects
    infosource = Node(IdentityInterface(fields = ['subject_id', 'run_list'], 
                                        run_list = run_list),
                      name = 'infosource')

    infosource.iterables = [('subject_id', subject_list)]

    
    func_file = opj(result_dir, output_dir, 'preprocessing', '_run_id_*_subject_id_{subject_id}', 
                    'swuasub-{subject_id}_task-MGT_run-*_bold.nii')
    
    motion_corrected_file = opj(result_dir, output_dir, 'preprocessing', '_run_id_*_subject_id_{subject_id}', 
                    'uasub-{subject_id}_task-MGT_run-*_bold.nii')
    
    parameter_file = opj(result_dir, output_dir, 'preprocessing', '_run_id_*_subject_id_{subject_id}', 
                         'rp_asub-{subject_id}_task-MGT_run-*_bold.txt')
    
    wc2_file = opj(result_dir, output_dir, 'preprocessing', '_run_id_01_subject_id_{subject_id}', 
                         'wc2sub-{subject_id}_T1w.nii')
    
    event_file = opj('sub-{subject_id}', 'func', 
                     'sub-{subject_id}_task-MGT_run-*_events.tsv')

    template = {'func' : func_file, 'param' : parameter_file, 'event' : event_file, 'wc2' : wc2_file, 
               'motion_correction': motion_corrected_file}

    # SelectFiles node - to select necessary files
    selectfiles = Node(SelectFiles(template, base_directory=exp_dir), name = 'selectfiles')
    
    # DataSink Node - store the wanted results in the wanted repository
    datasink = Node(DataSink(base_directory=result_dir, container=output_dir), name='datasink')
    

    # Get Subject Info - get subject specific condition information
    subject_infos = Node(Function(input_names=['event_files', 'runs'],
                                   output_names=['subject_info'],
                                   function=get_subject_infos),
                          name='subject_infos')
    
    # Get parameters 
    # compute_parameters(parameters_files, wc2_file, functional_files, subject_id, result_dir, working_dir)
    
    parameters = Node(Function(input_names = ['parameters_files', 'wc2_file', 'motion_corrected_files', 'subject_id', 
                                              'result_dir', 'working_dir'], 
                              output_names = ['new_parameters_files'],
                              function = compute_parameters), name = 'parameters')
    
    parameters.inputs.working_dir = working_dir
    parameters.inputs.result_dir = result_dir
    

    # SpecifyModel - Generates SPM-specific Model
    specify_model = Node(SpecifySPMModel(concatenate_runs = False, input_units = 'secs', output_units = 'secs',
                                        time_repetition = TR, high_pass_filter_cutoff = 128), name='specify_model')

    # Level1Design - Generates an SPM design matrix
    l1_design = Node(Level1Design(bases = {'hrf': {'derivs': [1, 1]}}, timing_units = 'secs', 
                                    interscan_interval = TR), name='l1_design')

    # EstimateModel - estimate the parameters of the model
    l1_estimate = Node(EstimateModel(estimation_method={'Classical': 1}),
                          name="l1_estimate")

    # Node contrasts to get contrasts 
    contrasts = Node(Function(function=get_contrasts,
                              input_names=['subject_id'],
                              output_names=['contrasts']),
                     name='contrasts')


    # EstimateContrast - estimates contrasts
    contrast_estimate = Node(EstimateContrast(), name="contrast_estimate")

    
    # Create l1 analysis workflow and connect its nodes
    l1_analysis = Workflow(base_dir = opj(result_dir, working_dir), name = "l1_analysis")

    l1_analysis.connect([(infosource, selectfiles, [('subject_id', 'subject_id')]),
                        (infosource, subject_infos, [('run_list', 'runs')]),
                        (infosource, contrasts, [('subject_id', 'subject_id')]),
                        (infosource, parameters, [('subject_id', 'subject_id')]),
                        (subject_infos, specify_model, [('subject_info', 'subject_info')]),
                        (contrasts, contrast_estimate, [('contrasts', 'contrasts')]),
                        (selectfiles, subject_infos, [('event', 'event_files')]),
                        (selectfiles, parameters, [('motion_correction', 'motion_corrected_files'),
                                                  ('param', 'parameters_files'),
                                                  ('wc2', 'wc2_file')]),
                        (selectfiles, specify_model, [('func', 'functional_runs')]),
                        (parameters, specify_model, [('new_parameters_files', 
                                                      'realignment_parameters')]),
                        (specify_model, l1_design, [('session_info', 'session_info')]),
                        (l1_design, l1_estimate, [('spm_mat_file', 'spm_mat_file')]),
                        (l1_estimate, contrast_estimate, [('spm_mat_file', 'spm_mat_file'),
                                                          ('beta_images', 'beta_images'),
                                                          ('residual_image', 'residual_image')]),
                        (contrast_estimate, datasink, [('con_images', 'l1_analysis.@con_images'),
                                                                ('spmT_images', 'l1_analysis.@spmT_images'),
                                                                ('spm_mat_file', 'l1_analysis.@spm_mat_file')])
                        ])
    
    return l1_analysis


def get_subset_contrasts(file_list, method, subject_list, participants_file):
    ''' 
    Parameters :
    - file_list : original file list selected by selectfiles node 
    - subject_list : list of subject IDs that are in the wanted group for the analysis
    - participants_file: str, file containing participants characteristics
    - method: str, one of "equalRange", "equalIndifference" or "groupComp"
    
    This function return the file list containing only the files belonging to subject in the wanted group.
    '''
    equalIndifference_id = []
    equalRange_id = []
    equalIndifference_files = []
    equalRange_files = []

    with open(participants_file, 'rt') as f:
            next(f)  # skip the header
            
            for line in f:
                info = line.strip().split()
                
                if info[0][-3:] in subject_list and info[1] == "equalIndifference":
                    equalIndifference_id.append(info[0][-3:])
                elif info[0][-3:] in subject_list and info[1] == "equalRange":
                    equalRange_id.append(info[0][-3:])
    
    for file in file_list:
        sub_id = file.split('/')
        if sub_id[-2][-3:] in equalIndifference_id:
            equalIndifference_files.append(file)
        elif sub_id[-2][-3:] in equalRange_id:
            equalRange_files.append(file)
            
    return equalIndifference_id, equalRange_id, equalIndifference_files, equalRange_files 


def get_l2_analysis(subject_list, n_sub, contrast_list, method, exp_dir, result_dir, working_dir, output_dir):
    """
    Returns the 2nd level of analysis workflow.

    Parameters: 
        - exp_dir: str, directory where raw data are stored
        - result_dir: str, directory where results will be stored
        - working_dir: str, name of the sub-directory for intermediate results
        - output_dir: str, name of the sub-directory for final results
        - subject_list: list of str, list of subject for which you want to do the preprocessing
        - contrast_list: list of str, list of contrasts to analyze
        - method: one of "equalRange", "equalIndifference" or "groupComp"
        - n_sub: int, number of subject

    Returns: 
        - l2_analysis: Nipype WorkFlow 
    """
    # Infosource - a function free node to iterate over the list of subject names
    infosource_groupanalysis = Node(IdentityInterface(fields=['contrast_id', 'subjects'],
                                                      subjects = subject_list),
                      name="infosource_groupanalysis")

    infosource_groupanalysis.iterables = [('contrast_id', contrast_list)]

    # SelectFiles
    contrast_file = opj(result_dir, output_dir, 'l1_analysis', '_subject_id_*', "con_00{contrast_id}.nii")

    participants_file = opj(exp_dir, 'participants.tsv')

    templates = {'contrast' : contrast_file, 'participants' : participants_file}
    
    selectfiles_groupanalysis = Node(SelectFiles(templates, base_directory=result_dir, force_list= True),
                       name="selectfiles_groupanalysis")
    
    # Datasink node : to save important files 
    datasink_groupanalysis = Node(DataSink(base_directory = result_dir, container = output_dir), 
                                  name = 'datasink_groupanalysis')
    
    # Node to select subset of contrasts
    sub_contrasts = Node(Function(input_names = ['file_list', 'method', 'subject_list', 'participants_file'],
                                 output_names = ['equalIndifference_id', 'equalRange_id', 
                                                 'equalIndifference_files', 'equalRange_files'],
                                 function = get_subset_contrasts),
                        name = 'sub_contrasts')
    
    sub_contrasts.inputs.method = method

    ## Estimate model 
    estimate_model = Node(EstimateModel(estimation_method={'Classical':1}), name = "estimate_model")

    ## Estimate contrasts
    estimate_contrast = Node(EstimateContrast(group_contrast=True),
                             name = "estimate_contrast")

    ## Create thresholded maps 
    threshold = MapNode(Threshold(use_fwe_correction=False, height_threshold = 0.01,
                                 extent_fdr_p_threshold = 0.05, use_topo_fdr = False, force_activation = True), name = "threshold", 
                        iterfield = ["stat_image", "contrast_index"])
    

    l2_analysis = Workflow(base_dir = opj(result_dir, working_dir), name = f"l2_analysis_{method}_nsub_{n_sub}")

    l2_analysis.connect([(infosource_groupanalysis, selectfiles_groupanalysis, [('contrast_id', 'contrast_id')]),
        (infosource_groupanalysis, sub_contrasts, [('subjects', 'subject_list')]),
        (selectfiles_groupanalysis, sub_contrasts, [('contrast', 'file_list'),
                                                    ('participants', 'participants_file')]),
        (estimate_model, estimate_contrast, [('spm_mat_file', 'spm_mat_file'),
            ('residual_image', 'residual_image'),
            ('beta_images', 'beta_images')]),
        (estimate_contrast, threshold, [('spm_mat_file', 'spm_mat_file'),('spmT_images', 'stat_image')]),
        (threshold, datasink_groupanalysis, [('thresholded_map', f"l2_analysis_{method}_nsub_{n_sub}.@thresh")]),
        (estimate_model, datasink_groupanalysis, [('mask_image', f"l2_analysis_{method}_nsub_{n_sub}.@mask")]),
        (estimate_contrast, datasink_groupanalysis, [('spm_mat_file', f"l2_analysis_{method}_nsub_{n_sub}.@spm_mat"),
            ('spmT_images', f"l2_analysis_{method}_nsub_{n_sub}.@T"),
            ('con_images', f"l2_analysis_{method}_nsub_{n_sub}.@con")])])
    
    
    if method=='equalRange' or method=='equalIndifference':
        contrasts = [('Group', 'T', ['mean'], [1]), ('Group', 'T', ['mean'], [-1])] 
        ## Specify design matrix 
        one_sample_t_test_design = Node(OneSampleTTestDesign(), name = "one_sample_t_test_design")

        l2_analysis.connect([(sub_contrasts, one_sample_t_test_design, [(f"{method}_files", 'in_files')]),
            (one_sample_t_test_design, estimate_model, [('spm_mat_file', 'spm_mat_file')])])
        
        threshold.inputs.contrast_index = [1, 2]
        threshold.synchronize = True

    elif method == 'groupComp':
        contrasts = [('Eq range vs Eq indiff in loss', 'T', ['Group_{1}', 'Group_{2}'], [1, -1])]
        # Node for the design matrix
        two_sample_t_test_design = Node(TwoSampleTTestDesign(), 
                                        name = 'two_sample_t_test_design')

        l2_analysis.connect([(sub_contrasts, two_sample_t_test_design, [('equalRange_files', "group1_files"), 
            ('equalIndifference_files', 'group2_files')]),
            (two_sample_t_test_design, estimate_model, [("spm_mat_file", "spm_mat_file")])])
        
        threshold.inputs.contrast_index = [1]
        threshold.synchronize = True

    estimate_contrast.inputs.contrasts = contrasts

    return l2_analysis

def reorganize_results(result_dir, output_dir, n_sub, team_ID):
    """
    Reorganize the results to analyze them. 

    Parameters: 
        - result_dir: str, directory where results will be stored
        - output_dir: str, name of the sub-directory for final results
        - n_sub: int, number of subject used for analysis
        - team_ID: str, name of the team ID for which we reorganize files
    """
    from os.path import join as opj
    import os
    import shutil
    import gzip

    h1 = opj(result_dir, output_dir, f"l2_analysis_equalIndifference_nsub_{n_sub}", '_contrast_id_01')
    h2 = opj(result_dir, output_dir, f"l2_analysis_equalRange_nsub_{n_sub}", '_contrast_id_01')
    h3 = opj(result_dir, output_dir, f"l2_analysis_equalIndifference_nsub_{n_sub}", '_contrast_id_01')
    h4 = opj(result_dir, output_dir, f"l2_analysis_equalRange_nsub_{n_sub}", '_contrast_id_01')
    h5 = opj(result_dir, output_dir, f"l2_analysis_equalIndifference_nsub_{n_sub}", '_contrast_id_02')
    h6 = opj(result_dir, output_dir, f"l2_analysis_equalRange_nsub_{n_sub}", '_contrast_id_02')
    h7 = opj(result_dir, output_dir, f"l2_analysis_equalIndifference_nsub_{n_sub}", '_contrast_id_02')
    h8 = opj(result_dir, output_dir, f"l2_analysis_equalRange_nsub_{n_sub}", '_contrast_id_02')
    h9 = opj(result_dir, output_dir, f"l2_analysis_groupComp_nsub_{n_sub}", '_contrast_id_02')

    h = [h1, h2, h3, h4, h5, h6, h7, h8, h9]

    repro_unthresh = [opj(filename, "spmT_0002.nii") if i in [4, 5] else opj(filename, 
                     "spmT_0001.nii") for i, filename in enumerate(h)]

    repro_thresh = [opj(filename, "_threshold1", 
         "spmT_0002_thr.nii") if i in [4, 5] else opj(filename, 
          "_threshold0", "spmT_0001_thr.nii")  for i, filename in enumerate(h)]
    
    if not os.path.isdir(opj(result_dir, "NARPS-reproduction")):
        os.mkdir(opj(result_dir, "NARPS-reproduction"))
    
    for i, filename in enumerate(repro_unthresh):
        f_in = filename
        f_out = opj(result_dir, "NARPS-reproduction", f"team_{team_ID}_nsub_{n_sub}_hypo{i+1}_unthresholded.nii")
        shutil.copyfile(f_in, f_out)

    for i, filename in enumerate(repro_thresh):
        f_in = filename
        f_out = opj(result_dir, "NARPS-reproduction", f"team_{team_ID}_nsub_{n_sub}_hypo{i+1}_thresholded.nii")
        shutil.copyfile(f_in, f_out)
    
    print(f"Results files of team {team_ID} reorganized.")