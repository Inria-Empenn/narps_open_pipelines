#!/usr/bin/python
# coding: utf-8

""" Write the work of NARPS' team C88N using Nipype """

#from os import
from os.path import join

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

def get_subject_infos_gain(event_files):
    '''
    Create Bunchs for specifySPMModel.
    Here, the team wanted to concatenate runs and used RT (response time) for duration except for NoResponse trials
    for which the duration was set to 4. 
    Gain and loss amounts were used as parametric regressors. 
    
    Parameters :
    - event_files : list of files containing events information for each run 
    
    Returns :
    - subject_info : list of Bunch for 1st level analysis.
    '''
    from os.path import join as opj
    from nipype.interfaces.base import Bunch
    
    cond_names = ['trial']
    onset = {}
    duration = {}
    weights_gain = {}
    weights_loss = {}
    runs = ['01', '02', '03', '04']
        
    for r in range(len(runs)):  # Loop over number of runs.
        onset.update({s + '_run' + str(r+1) : [] for s in cond_names}) # creates dictionary items with empty lists
        duration.update({s + '_run' + str(r+1) : [] for s in cond_names}) 
        weights_gain.update({'gain_run' + str(r+1) : []})
        weights_loss.update({'loss_run' + str(r+1) : []})
    
    # subject_id = '001'
    # file = sub-001_func_sub-001_task-MGT_run-01_events.tsv
    for r, f_events in enumerate(event_files): 
        with open(f_events, 'rt') as f:
            next(f)  # skip the header
            
            for line in f:
                info = line.strip().split()
                
                for cond in cond_names:
                    val = cond + '_run' + str(r+1) # trial_run1
                    val_gain = 'gain_run' + str(r+1) # gain_run1
                    val_loss = 'loss_run' + str(r+1) # loss_run1
                    onset[val].append(float(info[0])) # onsets for trial_run1 
                    duration[val].append(float(0)) # durations for trial : 0
                    weights_gain[val_gain].append(float(info[2])) # weights gain for trial_run1
                    weights_loss[val_loss].append(float(info[3])) # weights loss for trial_run1

    # Bunching is done per run, i.e. trial_run1, trial_run2, etc.
    # But names must not have '_run1' etc because we concatenate runs 
    subject_info = []
    for r in range(len(runs)):

        cond = [c + '_run' + str(r+1) for c in cond_names]
        gain = 'gain_run' + str(r+1)
        loss = 'loss_run' + str(r+1)

        subject_info.insert(r,
                           Bunch(conditions=cond_names,
                                 onsets=[onset[k] for k in cond],
                                 durations=[duration[k] for k in cond],
                                 amplitudes=None,
                                 tmod=None,
                                 pmod=[Bunch(name=['loss', 'gain'],
                                             poly=[1, 1],
                                             param=[weights_loss[loss],
                                                   weights_gain[gain]])],
                                 regressor_names=None,
                                 regressors=None))

    return subject_info

def get_subject_infos_loss(event_files):
    '''
    Create Bunchs for specifySPMModel.
    Here, the team wanted to concatenate runs and used RT (response time) for duration except for NoResponse trials
    for which the duration was set to 4. 
    Gain and loss amounts were used as parametric regressors. 
    
    Parameters :
    - event_files : list of files containing events information for each run 
    
    Returns :
    - subject_info : list of Bunch for 1st level analysis.
    '''
    from os.path import join as opj
    from nipype.interfaces.base import Bunch
    
    cond_names = ['trial']
    onset = {}
    duration = {}
    weights_gain = {}
    weights_loss = {}
    runs = ['01', '02', '03', '04']
        
    for r in range(len(runs)):  # Loop over number of runs.
        onset.update({s + '_run' + str(r+1) : [] for s in cond_names}) # creates dictionary items with empty lists
        duration.update({s + '_run' + str(r+1) : [] for s in cond_names}) 
        weights_gain.update({'gain_run' + str(r+1) : []})
        weights_loss.update({'loss_run' + str(r+1) : []})
    
    # subject_id = '001'
    # file = sub-001_func_sub-001_task-MGT_run-01_events.tsv
    for r, f_events in enumerate(event_files): 
        with open(f_events, 'rt') as f:
            next(f)  # skip the header
            
            for line in f:
                info = line.strip().split()
                
                for cond in cond_names:
                    val = cond + '_run' + str(r+1) # trial_run1
                    val_gain = 'gain_run' + str(r+1) # gain_run1
                    val_loss = 'loss_run' + str(r+1) # loss_run1
                    onset[val].append(float(info[0])) # onsets for trial_run1 
                    duration[val].append(float(0)) # durations for trial : 0
                    weights_gain[val_gain].append(float(info[2])) # weights gain for trial_run1
                    weights_loss[val_loss].append(float(info[3])) # weights loss for trial_run1

    # Bunching is done per run, i.e. trial_run1, trial_run2, etc.
    # But names must not have '_run1' etc because we concatenate runs 
    subject_info = []
    for r in range(len(runs)):

        cond = [c + '_run' + str(r+1) for c in cond_names]
        gain = 'gain_run' + str(r+1)
        loss = 'loss_run' + str(r+1)

        subject_info.insert(r,
                           Bunch(conditions=cond_names,
                                 onsets=[onset[k] for k in cond],
                                 durations=[duration[k] for k in cond],
                                 amplitudes=None,
                                 tmod=None,
                                 pmod=[Bunch(name=['gain', 'loss'],
                                             poly=[1, 1],
                                             param=[weights_gain[gain], weights_loss[loss]])],
                                 regressor_names=None,
                                 regressors=None))

    return subject_info

def get_contrasts_gain(subject_id):
    '''
    Create the list of tuples that represents contrasts. 
    Each contrast is in the form : 
    (Name,Stat,[list of condition names],[weights on those conditions])

    Parameters:
    	- subject_id: str, ID of the subject 

    Returns:
    	- contrasts: list of tuples, list of contrasts to analyze
    '''
    # list of condition names     
    conditions = ['trialxloss^1', 'trialxgain^1']
    
    # create contrasts    
    effect_gain = ('effect_of_gain', 'T', conditions, [0, 1])
    
    # contrast list
    contrasts = [effect_gain]

    return contrasts

def get_contrasts_loss(subject_id):
    '''
    Create the list of tuples that represents contrasts. 
    Each contrast is in the form : 
    (Name,Stat,[list of condition names],[weights on those conditions])

    Parameters:
    	- subject_id: str, ID of the subject 

    Returns:
    	- contrasts: list of tuples, list of contrasts to analyze
    '''
    # list of condition names     
    conditions = ['trialxgain^1', 'trialxloss^1']
    
    # create contrasts    
    positive_effect_loss = ('positive_effect_of_loss', 'T', conditions, [0, 1])
    
    negative_effect_loss = ('negative_effect_of_loss', 'T', conditions, [0, -1])
    
    # contrast list
    contrasts = [positive_effect_loss, negative_effect_loss]

    return contrasts

def rm_gunzip_files(files, subject_id, result_dir, working_dir):
    import shutil
    from os.path import join as opj

    gunzip_dir = opj(result_dir, working_dir, 'l1_analysis', f"_subject_id_{subject_id}", 'gunzip_func')
    
    try:
        shutil.rmtree(gunzip_dir)
    except OSError as e:
        print(e)
    else:
        print("The directory is deleted successfully")
    
    return files

def rm_smoothed_files(files, subject_id, result_dir, working_dir):
    import shutil
    from os.path import join as opj

    smooth_dir = opj(result_dir, working_dir, 'l1_analysis', f"_subject_id_{subject_id}", 'smooth')
    
    try:
        shutil.rmtree(smooth_dir)
    except OSError as e:
        print(e)
    else:
        print("The directory is deleted successfully")
    
    return files

def get_l1_analysis(subject_list, TR, fwhm, run_list, exp_dir, result_dir, working_dir, output_dir):
	"""
	Returns the first level analysis workflow.

	Parameters: 
		- exp_dir: str, directory where raw data are stored
		- result_dir: str, directory where results will be stored
		- working_dir: str, name of the sub-directory for intermediate results
		- output_dir: str, name of the sub-directory for final results
		- subject_list: list of str, list of subject for which you want to do the analysis
		- run_list: list of str, list of runs for which you want to do the analysis 
		- fwhm: float, fwhm for smoothing step
		- TR: float, time repetition used during acquisition

	Returns: 
		- l1_analysis : Nipype WorkFlow 
	"""
	# Infosource Node - To iterate on subjects
	infosource = Node(IdentityInterface(fields = ['subject_id']), name = 'infosource')
	infosource.iterables = [('subject_id', subject_list)]

	# Templates to select files node
	func_file = opj('derivatives', 'fmriprep', 'sub-{subject_id}', 'func', 
	                'sub-{subject_id}_task-MGT_run-*_bold_space-MNI152NLin2009cAsym_preproc.nii.gz')

	event_file = opj('sub-{subject_id}', 'func', 
	                'sub-{subject_id}_task-MGT_run-*_events.tsv')

	template = {'event' : event_file, 'func' : func_file}

	# SelectFiles node - to select necessary files
	selectfiles = Node(SelectFiles(template, base_directory=exp_dir), name = 'selectfiles')

	# DataSink Node - store the wanted results in the wanted repository
	datasink = Node(DataSink(base_directory=result_dir, container=output_dir), name='datasink')

	# GUNZIP NODE : SPM do not use .nii.gz files
	gunzip_func = MapNode(Gunzip(), name = 'gunzip_func', iterfield = ['in_file'])

	## Smoothing node
	smooth = Node(Smooth(fwhm = fwhm), name = 'smooth')

	# Get Subject Info - get subject specific condition information
	subject_infos_gain = Node(Function(input_names=['event_files'],
	                               output_names=['subject_info'],
	                               function=get_subject_infos_gain),
	                      name='subject_infos_gain')

	subject_infos_loss = Node(Function(input_names=['event_files'],
	                               output_names=['subject_info'],
	                               function=get_subject_infos_loss),
	                      name='subject_infos_loss')


	# SpecifyModel - Generates SPM-specific Model
	specify_model_gain = Node(SpecifySPMModel(concatenate_runs = True, input_units = 'secs', output_units = 'secs',
	                                    time_repetition = TR, high_pass_filter_cutoff = 128), 
	                          name='specify_model_gain')

	specify_model_loss = Node(SpecifySPMModel(concatenate_runs = True, input_units = 'secs', output_units = 'secs',
	                                    time_repetition = TR, high_pass_filter_cutoff = 128), 
	                          name='specify_model_loss')

	# Level1Design - Generates an SPM design matrix
	l1_design_gain = Node(Level1Design(bases = {'hrf': {'derivs': [0, 0]}}, timing_units = 'secs', 
	                                interscan_interval = TR), name='l1_design_gain')

	l1_design_loss = Node(Level1Design(bases = {'hrf': {'derivs': [0, 0]}}, timing_units = 'secs', 
	                                interscan_interval = TR), name='l1_design_loss')

	# EstimateModel - estimate the parameters of the model
	l1_estimate_gain = Node(EstimateModel(estimation_method={'Classical': 1}),
	                      name="l1_estimate_gain")

	l1_estimate_loss = Node(EstimateModel(estimation_method={'Classical': 1}),
	                      name="l1_estimate_loss")

	# Node contrasts to get contrasts 
	contrasts_gain = Node(Function(function=get_contrasts_gain,
	                          input_names=['subject_id'],
	                          output_names=['contrasts']),
	                 name='contrasts_gain')

	contrasts_loss = Node(Function(function=get_contrasts_loss,
	                          input_names=['subject_id'],
	                          output_names=['contrasts']),
	                 name='contrasts_loss')

	# EstimateContrast - estimates contrasts
	contrast_estimate_gain = Node(EstimateContrast(), name="contrast_estimate_gain")

	contrast_estimate_loss = Node(EstimateContrast(), name="contrast_estimate_loss")

	remove_gunzip_files = Node(Function(input_names = ['files', 'subject_id', 'result_dir', 'working_dir'],
	                            output_names = ['files'],
	                            function = rm_gunzip_files), name = 'remove_gunzip_files')

	remove_gunzip_files.inputs.result_dir = result_dir
	remove_gunzip_files.inputs.working_dir = working_dir

	remove_smoothed_files = Node(Function(input_names = ['files', 'subject_id', 'result_dir', 'working_dir'],
	                                     output_names = ['files'],
	                                     function = rm_smoothed_files), name = 'remove_smoothed_files')

	remove_smoothed_files.inputs.result_dir = result_dir
	remove_smoothed_files.inputs.working_dir = working_dir

	# Create l1 analysis workflow and connect its nodes
	l1_analysis = Workflow(base_dir = opj(result_dir, working_dir), name = "l1_analysis")

	l1_analysis.connect([(infosource, selectfiles, [('subject_id', 'subject_id')]),
	                    (selectfiles, subject_infos_gain, [('event','event_files')]),
	                    (selectfiles, subject_infos_loss, [('event','event_files')]),
	                    (infosource, contrasts_gain, [('subject_id', 'subject_id')]),
	                    (infosource, contrasts_loss, [('subject_id', 'subject_id')]),
	                    (infosource, remove_gunzip_files, [('subject_id', 'subject_id')]),
	                    (infosource, remove_smoothed_files, [('subject_id', 'subject_id')]),
	                    (subject_infos_gain, specify_model_gain, [('subject_info', 'subject_info')]),
	                    (subject_infos_loss, specify_model_loss, [('subject_info', 'subject_info')]),
	                    (contrasts_gain, contrast_estimate_gain, [('contrasts', 'contrasts')]),
	                    (contrasts_loss, contrast_estimate_loss, [('contrasts', 'contrasts')]),
	                    (selectfiles, gunzip_func, [('func', 'in_file')]),
	                    (gunzip_func, smooth, [('out_file', 'in_files')]),
	                    (smooth, remove_gunzip_files, [('smoothed_files', 'files')]),
	                    (remove_gunzip_files, specify_model_gain, [('files', 'functional_runs')]),
	                    (remove_gunzip_files, specify_model_loss, [('files', 'functional_runs')]),
	                    (specify_model_gain, l1_design_gain, [('session_info', 'session_info')]),
	                    (specify_model_loss, l1_design_loss, [('session_info', 'session_info')]),
	                    (l1_design_gain, l1_estimate_gain, [('spm_mat_file', 'spm_mat_file')]),
	                    (l1_design_loss, l1_estimate_loss, [('spm_mat_file', 'spm_mat_file')]),
	                    (l1_estimate_gain, contrast_estimate_gain, [('spm_mat_file', 'spm_mat_file'),
	                                                      ('beta_images', 'beta_images'),
	                                                      ('residual_image', 'residual_image')]),
	                    (l1_estimate_loss, contrast_estimate_loss, [('spm_mat_file', 'spm_mat_file'),
	                                                      ('beta_images', 'beta_images'),
	                                                      ('residual_image', 'residual_image')]),
	                    (contrast_estimate_gain, datasink, [('con_images', 'l1_analysis_gain.@con_images'),
	                                                            ('spmT_images', 'l1_analysis_gain.@spmT_images'),
	                                                            ('spm_mat_file', 'l1_analysis_gain.@spm_mat_file')]),
	                    (contrast_estimate_loss, datasink, [('con_images', 'l1_analysis_loss.@con_images'),
	                                                            ('spmT_images', 'l1_analysis_loss.@spmT_images'),
	                                                            ('spm_mat_file', 'l1_analysis_loss.@spm_mat_file')]),
	                    (contrast_estimate_gain, remove_smoothed_files, [('spmT_images', 'files')])
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


def get_l2_analysis(subject_list, n_sub, model_list, contrast_list, method, exp_dir, result_dir, working_dir, output_dir):
    """
    Returns the 2nd level of analysis workflow.

    Parameters: 
        - exp_dir: str, directory where raw data are stored
        - result_dir: str, directory where results will be stored
        - working_dir: str, name of the sub-directory for intermediate results
        - output_dir: str, name of the sub-directory for final results
        - subject_list: list of str, list of subject for which you want to do the preprocessing
        - model_list: list of str, list of models to use for the analysis
        - contrast_list: list of str, list of contrasts to analyze
        - n_sub: float, number of subjects used to do the analysis
        - method: one of "equalRange", "equalIndifference" or "groupComp"

    Returns: 
        - l2_analysis: Nipype WorkFlow 
    """      
    # Infosource - a function free node to iterate over the list of subject names
    infosource_groupanalysis = Node(IdentityInterface(fields=['model_type','contrast_id', 'subjects'],
                                                      subjects = subject_list),
                      name="infosource_groupanalysis")

    infosource_groupanalysis.iterables = [('model_type', model_list), ('contrast_id', contrast_list)]

    # SelectFiles
    contrast_file = opj(result_dir, output_dir, "l1_analysis_{model_type}", "_subject_id_*", "con_000{contrast_id}.nii")

    participants_file = opj(exp_dir,  'participants.tsv')

    templates = {'contrast' : contrast_file, 'participants' : participants_file}
    
    selectfiles_groupanalysis = Node(SelectFiles(templates, base_directory=result_dir, force_list= True),
                       name="selectfiles_groupanalysis")
    
    # Datasink node : to save important files 
    datasink_groupanalysis = Node(DataSink(base_directory = result_dir, container = output_dir), 
                                  name = 'datasink_groupanalysis')
    
    # Node to select subset of contrasts
    sub_contrasts = Node(Function(input_names = ['file_list', 'method', 'subject_list', 'participants_file'],
                                 output_names = ['equalIndifference_id', 'equalRange_id', 'equalIndifference_files', 'equalRange_files'],
                                 function = get_subset_contrasts),
                        name = 'sub_contrasts')

    sub_contrasts.inputs.method = method

    ## Estimate model 
    estimate_model = Node(EstimateModel(estimation_method={'Classical':1}), name = "estimate_model")

    ## Estimate contrasts
    estimate_contrast = Node(EstimateContrast(group_contrast=True),
                             name = "estimate_contrast")

    ## Create thresholded maps 
    threshold = MapNode(Threshold(contrast_index=1,
                              use_topo_fdr=True,
                              use_fwe_correction=False,
                              extent_threshold=0,
                              height_threshold=0.001,
                              height_threshold_type='p-value'), name = "threshold", 
                        iterfield = ["stat_image", "contrast_index"])


    l2_analysis = Workflow(base_dir = opj(result_dir, working_dir), name = 'l2_analysis')

    l2_analysis.connect([(infosource_groupanalysis, selectfiles_groupanalysis, [('contrast_id', 'contrast_id'), 
                                                                                ('model_type', 'model_type')]),
        (infosource_groupanalysis, sub_contrasts, [('subjects', 'subject_list')]),
        (selectfiles_groupanalysis, sub_contrasts, [('contrast', 'file_list'), ('participants', 'participants_file')]),
        (estimate_model, estimate_contrast, [('spm_mat_file', 'spm_mat_file'),
            ('residual_image', 'residual_image'),
            ('beta_images', 'beta_images')]),
        (estimate_contrast, threshold, [('spm_mat_file', 'spm_mat_file'),
            ('spmT_images', 'stat_image')]),
        (estimate_model, datasink_groupanalysis, [('mask_image', f"l2_analysis_{method}_nsub_{n_sub}.@mask")]),
        (estimate_contrast, datasink_groupanalysis, [('spm_mat_file', f"l2_analysis_{method}_nsub_{n_sub}.@spm_mat"),
            ('spmT_images', f"l2_analysis_{method}_nsub_{n_sub}.@T"),
            ('con_images', f"l2_analysis_{method}_nsub_{n_sub}.@con")]),
        (threshold, datasink_groupanalysis, [('thresholded_map', f"l2_analysis_{method}_nsub_{n_sub}.@thresh")])])
    
    if method=='equalRange' or method=='equalIndifference':
        contrasts = [('Group', 'T', ['mean'], [1]), ('Group', 'T', ['mean'], [-1])] 
        ## Specify design matrix 
        one_sample_t_test_design = Node(OneSampleTTestDesign(), name = "one_sample_t_test_design")

        l2_analysis.connect([(sub_contrasts, one_sample_t_test_design, [(f"{method}_files", 'in_files')]),
            (one_sample_t_test_design, estimate_model, [('spm_mat_file', 'spm_mat_file')])])
        
        threshold.inputs.contrast_index = [1, 2]
        threshold.synchronize = True

        l2_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = f'l2_analysis_{method}_nsub_{nb_subjects}')
        l2_analysis.connect([
            (infosource_groupanalysis, selectfiles_groupanalysis, [
                ('contrast_id', 'contrast_id'),
                ('model_type', 'model_type')]),
            (infosource_groupanalysis, sub_contrasts, [
                ('subjects', 'subject_list')]),
            (selectfiles_groupanalysis, sub_contrasts, [
                ('contrasts', 'file_list'),
                ('participants', 'participants_file')]),
            (estimate_model, estimate_contrast, [
                ('spm_mat_file', 'spm_mat_file'),
                ('residual_image', 'residual_image'),
                ('beta_images', 'beta_images')]),
            (estimate_contrast, threshold, [
                ('spm_mat_file', 'spm_mat_file'),
                ('spmT_images', 'stat_image')]),
            (estimate_model, datasink_groupanalysis, [
                ('mask_image', f'l2_analysis_{method}_nsub_{nb_subjects}.@mask')]),
            (estimate_contrast, datasink_groupanalysis, [
                ('spm_mat_file', f'l2_analysis_{method}_nsub_{nb_subjects}.@spm_mat'),
                ('spmT_images', f'l2_analysis_{method}_nsub_{nb_subjects}.@T'),
                ('con_images', f'l2_analysis_{method}_nsub_{nb_subjects}.@con')]),
            (threshold, datasink_groupanalysis, [
                ('thresholded_map', f'l2_analysis_{method}_nsub_{nb_subjects}.@thresh')])])

        if method in ('equalRange', 'equalIndifference'):
            contrasts = [('Group', 'T', ['mean'], [1]), ('Group', 'T', ['mean'], [-1])]
            threshold.inputs.contrast_index = [1, 2]

            # Specify design matrix
            one_sample_t_test_design = Node(OneSampleTTestDesign(),
                name = 'one_sample_t_test_design')

            l2_analysis.connect([
                (sub_contrasts, one_sample_t_test_design, [(f'{method}_files', 'in_files')]),
                (one_sample_t_test_design, estimate_model, [('spm_mat_file', 'spm_mat_file')])
                ])

        elif method == 'groupComp':
            contrasts = [
                ('Eq range vs Eq indiff in loss', 'T', ['Group_{1}', 'Group_{2}'], [1, -1])]
            threshold.inputs.contrast_index = [1]

            # Specify design matrix
            two_sample_t_test_design = Node(TwoSampleTTestDesign(),
                name = 'two_sample_t_test_design')

            l2_analysis.connect([
                (sub_contrasts, two_sample_t_test_design, [
                    ('equalRange_files', 'group1_files'),
                    ('equalIndifference_files', 'group2_files')]),
                (two_sample_t_test_design, estimate_model, [
                    ('spm_mat_file', 'spm_mat_file')])
                ])

        estimate_contrast.inputs.contrasts = contrasts

        return l2_analysis
