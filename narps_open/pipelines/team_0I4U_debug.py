from nipype.interfaces.spm import (Coregister, Smooth, OneSampleTTestDesign, EstimateModel, EstimateContrast, 
                                   Level1Design, TwoSampleTTestDesign, RealignUnwarp, FieldMap, NewSegment, 
                                   Normalize12, Reslice)
from nipype.interfaces.spm import Threshold 
from nipype.interfaces.fsl import ApplyMask, ExtractROI, ImageMaths
from nipype.algorithms.modelgen import SpecifySPMModel
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.algorithms.misc import Gunzip
from nipype import Workflow, Node, MapNode, JoinNode
from nipype.interfaces.base import Bunch

from os.path import join as opj
import os
import json

def rm_preproc_files(files, run_id, subject_id, result_dir, working_dir):
    import shutil
    from os.path import join as opj
    import os

    preproc_dir = opj(result_dir, working_dir, 'preprocessing', f"_run_id_{run_id}_subject_id_{subject_id}")
    dir_to_rm = ['coregistration']
    for dirs in dir_to_rm:
        try:
            shutil.rmtree(opj(preproc_dir, dirs))
        except OSError as e:
            print(e)
        else:
            print(f"The directory {dirs} is deleted successfully")
    
    return files

def rm_gunzip_files(files, run_id, subject_id, result_dir, working_dir):
    import shutil
    from os.path import join as opj

    preproc_dir = opj(result_dir, working_dir, 'preprocessing', f"_run_id_{run_id}_subject_id_{subject_id}")

    dir_to_rm = ['gunzip_func', 'gunzip_magnitude', 'gunzip_phasediff', 'fieldmap', 'extract_first']
    for dirs in dir_to_rm:
        try:
            shutil.rmtree(opj(preproc_dir, dirs))
        except OSError as e:
            print(e)
        else:
            print(f"The directory {dirs} is deleted successfully")

    return files    

def get_preprocessing(exp_dir, result_dir, working_dir, output_dir, subject_list, run_list, fwhm, TR, 
                      total_readout_time):
    """
    Returns the preprocessing workflow.

    Parameters: 
        - exp_dir: str, directory where raw data are stored
        - result_dir: str, directory where results will be stored
        - working_dir: str, name of the sub-directory for intermediate results
        - output_dir: str, name of the sub-directory for final results
        - subject_list: list of str, list of subject for which you want to do the preprocessing
        - run_list: list of str, list of runs for which you want to do the preprocessing 
        - fwhm: float, fwhm for smoothing step
        - TR: float, time repetition used during acquisition
        - total_readout_time: float, time taken to acquire all of the phase encode steps required to cover k-space (i.e., one image slice)

    Returns: 
        - preprocessing: Nipype WorkFlow 
    """
    # Templates to select files node
    infosource_preproc = Node(IdentityInterface(fields = ['subject_id', 'run_id']), 
        name = 'infosource_preproc')

    infosource_preproc.iterables = [('subject_id', subject_list), ('run_id', run_list)]

    # Templates to select files node
    anat_file = opj('sub-{subject_id}', 'anat', 
                    'sub-{subject_id}_T1w.nii.gz')

    func_file = opj('sub-{subject_id}', 'func', 
                    'sub-{subject_id}_task-MGT_run-{run_id}_bold.nii.gz')
    
    magnitude_file = opj('sub-{subject_id}', 'fmap', 'sub-{subject_id}_magnitude1.nii.gz')

    phasediff_file = opj('sub-{subject_id}', 'fmap', 'sub-{subject_id}_phasediff.nii.gz')
    
    template = {'anat' : anat_file, 'func' : func_file, 'magnitude' : magnitude_file, 'phasediff' : phasediff_file}

    # SelectFiles node - to select necessary files
    selectfiles_preproc = Node(SelectFiles(template, base_directory=exp_dir), name = 'selectfiles_preproc')

    # GUNZIP NODE : SPM do not use .nii.gz files
    gunzip_anat = Node(Gunzip(), name = 'gunzip_anat')

    gunzip_func = Node(Gunzip(), name = 'gunzip_func')

    gunzip_magnitude = Node(Gunzip(), name = 'gunzip_magnitude')

    gunzip_phasediff = Node(Gunzip(), name = 'gunzip_phasediff')

    fieldmap = Node(FieldMap(blip_direction = -1, echo_times=(4.92, 7.38), 
                            total_readout_time = 29.15, matchanat=True,
                            matchvdm=True, writeunwarped = True, maskbrain = False, thresh = 0), name = 'fieldmap')

    motion_correction = Node(RealignUnwarp(interp = 4, register_to_mean = True),
                             name = 'motion_correction')
    
    extract_epi = Node(ExtractROI(t_min = 1, t_size=1, output_type = 'NIFTI'), name = 'extract_ROI')
    
    extract_first = Node(ExtractROI(t_min = 1, t_size=1, output_type = 'NIFTI'), name = 'extract_first')

    coregistration = Node(Coregister(jobtype = 'estimate', write_mask = False), name = 'coregistration')
    
    tissue1 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 1), 1, (True,False), (True, False)]
    tissue2 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 2), 1, (True,False), (True, False)]
    tissue3 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 3), 2, (True,False), (True, False)]
    tissue4 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 4), 3, (True,False), (True, False)]
    tissue5 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 5), 4, (True,False), (True, False)]
    tissue6 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 6), 2, (True,False), (True, False)]
    tissue_list = [tissue1, tissue2, tissue3, tissue4, tissue5, tissue6]
    
    segmentation = Node(NewSegment(write_deformation_fields = [True, True], tissues = tissue_list), name = 'segmentation')
    
    normalize = Node(Normalize12(write_voxel_sizes=[1.5, 1.5, 1.5], jobtype = 'write'), name = 'normalize')

    rm_files = Node(Function(input_names = ['files', 'run_id', 'subject_id', 'result_dir', 'working_dir'],
                            output_names = ['files'], function = rm_preproc_files), name = 'rm_files')

    rm_files.inputs.result_dir = result_dir
    rm_files.inputs.working_dir = working_dir

    rm_gunzip = Node(Function(input_names = ['files', 'run_id', 'subject_id', 'result_dir', 'working_dir'],
                            output_names = ['files'], function = rm_gunzip_files), name = 'rm_gunzip')

    rm_gunzip.inputs.result_dir = result_dir
    rm_gunzip.inputs.working_dir = working_dir

    # DataSink Node - store the wanted results in the wanted repository
    datasink_preproc = Node(DataSink(base_directory=result_dir, container=output_dir), name='datasink_preproc')

    preprocessing =  Workflow(base_dir = opj(result_dir, working_dir), name = "preprocessing")

    preprocessing.connect([(infosource_preproc, selectfiles_preproc, [('subject_id', 'subject_id'),
                                                                     ('run_id', 'run_id')]), 
                          (infosource_preproc, rm_files, [('subject_id', 'subject_id'), ('run_id', 'run_id')]),
                          (infosource_preproc, rm_gunzip, [('subject_id', 'subject_id'), ('run_id', 'run_id')]),
                          (selectfiles_preproc, gunzip_anat, [('anat', 'in_file')]),
                          (selectfiles_preproc, gunzip_func, [('func', 'in_file')]),
                          (selectfiles_preproc, gunzip_phasediff, [('phasediff', 'in_file')]),
                          (selectfiles_preproc, gunzip_magnitude, [('magnitude', 'in_file')]),
                          (gunzip_anat, segmentation, [('out_file', 'channel_files')]),
                          (gunzip_anat, fieldmap, [('out_file', 'anat_file')]),
                          (gunzip_magnitude, fieldmap, [('out_file', 'magnitude_file')]),
                          (gunzip_phasediff, fieldmap, [('out_file', 'phase_file')]),
                          (gunzip_func, extract_first, [('out_file', 'in_file')]),
                          (extract_first, fieldmap, [('roi_file', 'epi_file')]),
                          (fieldmap, motion_correction, [('vdm', 'phase_map')]),
                          (gunzip_func, motion_correction, [('out_file', 'in_files')]), 
                          (motion_correction, rm_gunzip, [('realigned_unwarped_files', 'files')]),
                          (rm_gunzip, coregistration, [('files', 'apply_to_files')]),
                          (motion_correction, coregistration, [('mean_image', 'source')]),
                          (coregistration, normalize, [('coregistered_files', 'apply_to_files')]),
                          (gunzip_anat, coregistration, [('out_file', 'target')]),
                          (segmentation, normalize, [('forward_deformation_field', 'deformation_file')]),
                          (normalize, rm_files, [('normalized_files', 'files')]), 
                          (rm_files, datasink_preproc, [('files', 'preprocess.@norm')]),
                          (motion_correction, datasink_preproc, [('realignment_parameters', 
                                                                  'preprocess.@realign_par')]),
                          (segmentation, datasink_preproc, [('normalized_class_images', 'preprocess.@gm')])
                          ])
    
    return preprocessing


def get_subject_infos(event_files, runs):
    '''
    Event-related design, 4 within subject sessions
    1 Condition: Stimulus presentation, onsets based on tsv file, duration 4 seconds
    2 Parametric modulators: Gain and loss modelled with 1st order polynomial expansion
    1 Condition: button press, onsets based on tsv file, duration 0 seconds

    Create Bunchs for specifySPMModel.

    Parameters :
    - event_files: list of str, list of events files (one per run) for the subject
    - runs: list of str, list of runs to use
    
    Returns :
    - subject_info : list of Bunch for 1st level analysis.
    '''
    from nipype.interfaces.base import Bunch
    
    cond_names = ['trial']
    onset = {}
    duration = {}
    weights_gain = {}
    weights_loss = {}
    onset_button = {}
    duration_button = {}
    
    for r in range(len(runs)):  # Loop over number of runs.
        onset.update({s + '_run' + str(r+1) : [] for s in cond_names}) # creates dictionary items with empty lists
        duration.update({s + '_run' + str(r+1) : [] for s in cond_names}) 
        weights_gain.update({'gain_run' + str(r+1) : []})
        weights_loss.update({'loss_run' + str(r+1) : []})
        onset_button.update({'button_run' + str(r+1) : []})
        duration_button.update({'button_run' + str(r+1) : []})
    
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
                    val_button = 'button_run' + str(r+1)
                    onset[val].append(float(info[0])) # onsets for trial_run1 
                    duration[val].append(float(4))
                    weights_gain[val_gain].append(float(info[2])) # weights gain for trial_run1
                    weights_loss[val_loss].append(float(info[3])) # weights loss for trial_run1
                    onset_button[val_button].append(float(info[0]) + float(info[4]))
                    duration_button[val_button].append(float(0))

    # Bunching is done per run, i.e. trial_run1, trial_run2, etc.
    # But names must not have '_run1' etc because we concatenate runs 
    subject_info = []
    for r in range(len(runs)):

        cond = 'trial_run' + str(r+1)
        button_cond = 'button_run' + str(r+1)
        gain = 'gain_run' + str(r+1)
        loss = 'loss_run' + str(r+1)

        subject_info.insert(r,
                           Bunch(conditions=[cond],
                                 onsets=[onset[cond]],
                                 durations=[duration[cond]],
                                 amplitudes=None,
                                 tmod=None,
                                 pmod=[Bunch(name=[gain, loss],
                                             poly=[1, 1],
                                             param=[weights_gain[gain],
                                                    weights_loss[loss]]), None],
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

    gain = []
    loss = []
    for ir in range(runs):
        ir += 1
        gain.append('trial_run%ixgain_run%i^1' % (ir, ir))
        loss.append('trial_run%ixloss_run%i^1' % (ir, ir))

    pos_1 = [1] * runs

    gain = (
        'gain', 'T',
        gain, pos_1)

    loss = (
        'loss', 'T',
        loss, pos_1)

    contrasts = [gain, loss]

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
    infosource = Node(IdentityInterface(fields = ['subject_id', 'exp_dir', 'result_dir', 
                                                  'working_dir', 'run_list'], 
                                        exp_dir = exp_dir, result_dir = result_dir, working_dir = working_dir,
                                        run_list = run_list),
                      name = 'infosource')

    infosource.iterables = [('subject_id', subject_list)]

    # Templates to select files node
    param_file = opj(result_dir, output_dir, 'preprocess', '_run_id_*_subject_id_{subject_id}', 
                    'rp_sub-{subject_id}_task-MGT_run-*_bold.txt')

    func_file = opj(result_dir, output_dir, 'preprocess', '_run_id_*_subject_id_{subject_id}', 
                   'wusub-{subject_id}_task-MGT_run-*_bold.nii')

    event_files = opj(exp_dir, 'sub-{subject_id}', 'func', 
                      'sub-{subject_id}_task-MGT_run-*_events.tsv')

    template = {'param' : param_file, 'func' : func_file, 'event' : event_files, 'mask':mask_file}

    # SelectFiles node - to select necessary files
    selectfiles = Node(SelectFiles(template, base_directory=exp_dir, force_list= True), name = 'selectfiles')
    
    # DataSink Node - store the wanted results in the wanted repository
    datasink = Node(DataSink(base_directory=result_dir, container=output_dir), name='datasink')
    
    # Smoothing
    smooth = Node(Smooth(fwhm = 5), name = 'smooth')

    # Get Subject Info - get subject specific condition information
    subject_infos = Node(Function(input_names=['event_files', 'runs'],
                                   output_names=['subject_info'],
                                   function=get_subject_infos),
                          name='subject_infos')
    
    subject_infos.inputs.runs = run_list
    
    # SpecifyModel - Generates SPM-specific Model
    specify_model = Node(SpecifySPMModel(concatenate_runs = False, input_units = 'secs', output_units = 'secs',
                                        time_repetition = TR, high_pass_filter_cutoff = 128), name='specify_model')

    # Level1Design - Generates an SPM design matrix
    l1_design = Node(Level1Design(bases = {'hrf': {'derivs': [1, 0]}}, timing_units = 'secs', 
                                    interscan_interval = TR, 
                                 model_serial_correlations = 'AR(1)'),
                     name='l1_design')

    # EstimateModel - estimate the parameters of the model
    l1_estimate = Node(EstimateModel(estimation_method={'Classical': 1}, write_residuals = False),
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
                        (infosource, contrasts, [('subject_id', 'subject_id')]),
                        (subject_infos, specify_model, [('subject_info', 'subject_info')]),
                        (contrasts, contrast_estimate, [('contrasts', 'contrasts')]),
                        (selectfiles, subject_infos, [('event', 'event_files')]),
                        (selectfiles, specify_model, [('param', 'realignment_parameters')]),
                        (selectfiles, smooth, [('func', 'in_files')]),
                        (smooth, specify_model, [('smoothed_files', 'functional_runs')]),
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

def get_subset_contrasts(file_list, subject_list, participants_file, method):
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
    equalRange_covar_val = [[], []]
    equalIndifference_covar_val = [[], []]
    """
    covariates (a list of items which are a dictionary with keys which are ‘vector’ or ‘name’ or ‘interaction’ 
    or ‘centering’ and with values which are any value) – Covariate dictionary {vector, name, interaction,
    centering}.
    """

    with open(participants_file, 'rt') as f:
            next(f)  # skip the header
            
            for line in f:
                info = line.strip().split()
                
                if info[0][-3:] in subject_list and info[1] == "equalIndifference":
                    equalIndifference_id.append(info[0][-3:])
                    equalIndifference_covar_val[0].append(float(info[3]))
                    equalIndifference_covar_val[1].append(0 if info[2]=='M' else 1)
                elif info[0][-3:] in subject_list and info[1] == "equalRange":
                    equalRange_id.append(info[0][-3:])
                    equalRange_covar_val[0].append(float(info[3]))
                    equalRange_covar_val[1].append(0 if info[2]=='M' else 1)
    
    for file in file_list:
        sub_id = file.split('/')
        if sub_id[-2][-3:] in equalIndifference_id:
            equalIndifference_files.append(file)
        elif sub_id[-2][-3:] in equalRange_id:
            equalRange_files.append(file)
            
    equalRange_covar = [dict(vector=equalRange_covar_val[0], name='age', centering=[1]),
                        dict(vector=equalRange_covar_val[1], name='sex', centering=[1])]
    
    equalIndifference_covar = [dict(vector=equalIndifference_covar_val[0], name='age', centering=[1]),
                               dict(vector=equalIndifference_covar_val[1], name='sex', centering=[1])]
    
    nb_eqRange = len(equalRange_id)
    nb_eqIndifference = len(equalIndifference_id)
    
    global_covar = []
    
    if method == 'groupComp':
        global_covar = [dict(vector=equalRange_covar_val[0], name='eqRange_age'),
                        dict(vector=equalIndifference_covar_val[0], name='eqIndifference_age'),
                        dict(vector=equalRange_covar_val[1], name='eqRange_sex'),
                        dict(vector=equalIndifference_covar_val[1], name='eqIndifference_sex')]

        for i in range(4):
            if i in [0, 2]:
                for k in range(nb_eqIndifference):
                    global_covar[i]['vector'].append(0)
            elif i in [1, 3]:
                for k in range(nb_eqRange):
                    global_covar[i]['vector'].insert(0 ,0)
    
            
    return equalIndifference_id, equalRange_id, equalIndifference_files, equalRange_files, equalRange_covar, equalIndifference_covar, global_covar 


def get_l2_analysis(subject_list, n_sub, contrast_list, method, exp_dir, result_dir, working_dir, output_dir, data_dir):
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
        - nsub: int, number of subject in subject_list

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
    
    mask_file = opj(data_dir, 'NARPS-0I4U', 'hypo1_unthresh.nii.gz')

    templates = {'contrast' : contrast_file, 'participants' : participants_file, 'mask':mask_file}
    
    selectfiles_groupanalysis = Node(SelectFiles(templates, base_directory=result_dir, force_list= True),
                       name="selectfiles_groupanalysis")
    
    # Datasink node : to save important files 
    datasink_groupanalysis = Node(DataSink(base_directory = result_dir, container = output_dir), 
                                  name = 'datasink_groupanalysis')
    
    gunzip_mask = Node(Gunzip(), name='gunzip_mask')
    # Node to select subset of contrasts
    sub_contrasts = Node(Function(input_names = ['file_list', 'subject_list', 'participants_file', 'method'],
                                 output_names = ['equalIndifference_id', 'equalRange_id', 
                                                 'equalIndifference_files', 'equalRange_files',
                                                'equalRange_covar', 'equalIndifference_covar', 'global_covar'],
                                 function = get_subset_contrasts),
                        name = 'sub_contrasts')
    
    sub_contrasts.inputs.method = method

    ## Estimate model 
    estimate_model = Node(EstimateModel(estimation_method={'Classical':1}), name = "estimate_model")

    ## Estimate contrasts
    estimate_contrast = Node(EstimateContrast(group_contrast=True),
                             name = "estimate_contrast")

    ## Create thresholded maps 
    threshold = MapNode(Threshold(use_fwe_correction=True,
                                  height_threshold_type='p-value',
                                  force_activation = False), name = "threshold", 
                        iterfield = ["stat_image", "contrast_index"])
    

    l2_analysis = Workflow(base_dir = opj(result_dir, working_dir), name = f"l2_analysis_{method}_nsub_{n_sub}")

    l2_analysis.connect([(infosource_groupanalysis, selectfiles_groupanalysis, [('contrast_id', 'contrast_id')]),
        (infosource_groupanalysis, sub_contrasts, [('subjects', 'subject_list')]),
        (selectfiles_groupanalysis, sub_contrasts, [('contrast', 'file_list'),
                                                    ('participants', 'participants_file')]),
        (selectfiles_groupanalysis, gunzip_mask, [('mask', 'in_file')]),
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
        one_sample_t_test_design = Node(OneSampleTTestDesign(use_implicit_threshold=True), name = "one_sample_t_test_design")

        l2_analysis.connect([(sub_contrasts, one_sample_t_test_design, [(f"{method}_files", 'in_files'), 
                                                                       (f"{method}_covar", "covariates")]),
                             (gunzip_mask, one_sample_t_test_design, [('out_file', 'explicit_mask_file')]),
            (one_sample_t_test_design, estimate_model, [('spm_mat_file', 'spm_mat_file')])])
        
        threshold.inputs.contrast_index = [1, 2]
        threshold.synchronize = True

    elif method == 'groupComp':
        contrasts = [('Eq range vs Eq indiff in loss', 'T', ['Group_{1}', 'Group_{2}'], [1, -1])]
        # Node for the design matrix
        two_sample_t_test_design = Node(TwoSampleTTestDesign(unequal_variance = True, use_implicit_threshold=True), 
                                        name = 'two_sample_t_test_design')

        l2_analysis.connect([(sub_contrasts, two_sample_t_test_design, [('equalRange_files', "group1_files"), 
            ('equalIndifference_files', 'group2_files'), ('global_covar', 'covariates')]),
            (gunzip_mask, two_sample_t_test_design, [('out_file', 'explicit_mask_file')]),
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
         "spmT_0002_thr.nii") if i in [6, 7] else opj(filename, 
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
