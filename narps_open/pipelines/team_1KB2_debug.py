from nipype.interfaces.fsl import (BET, FAST, MCFLIRT, FLIRT, FNIRT, ApplyWarp, SUSAN, 
                                   Info, ImageMaths, IsotropicSmooth, Threshold, Level1Design, FEATModel, 
                                   L2Model, Merge, FLAMEO, ContrastMgr,Cluster,  FILMGLS, Randomise, MultipleRegressDesign)
from nipype.algorithms.modelgen import SpecifyModel

from niflow.nipype1.workflows.fmri.fsl import create_susan_smooth
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.algorithms.misc import Gunzip
from nipype import Workflow, Node, MapNode
from nipype.interfaces.base import Bunch

from os.path import join as opj
import os

def get_preprocessing_1st_step(exp_dir, result_dir, working_dir, output_dir, subject_list, run_list, fwhm):
    """
    Returns the 1st step of the preprocessing workflow.

    Parameters: 
        - exp_dir: str, directory where raw data are stored
        - result_dir: str, directory where results will be stored
        - working_dir: str, name of the sub-directory for intermediate results
        - output_dir: str, name of the sub-directory for final results
        - subject_list: list of str, list of subject for which you want to do the preprocessing
        - run_list: list of str, list of runs for which you want to do the preprocessing 
        - fwhm: float, fwhm for smoothing step
        
    Returns: 
        - preprocessing: Nipype WorkFlow 
    """
    infosource_preproc = Node(IdentityInterface(fields = ['subject_id', 'run_id']), 
        name = 'infosource_preproc')

    infosource_preproc.iterables = [('subject_id', subject_list), ('run_id', run_list)]

    # Templates to select files node
    func_file = opj('sub-{subject_id}', 'func', 
                    'sub-{subject_id}_task-MGT_run-{run_id}_bold.nii.gz')

    template = {'func' : func_file}

    # SelectFiles node - to select necessary files
    selectfiles_preproc = Node(SelectFiles(template, base_directory=exp_dir), name = 'selectfiles_preproc')


    motion_correction = Node(MCFLIRT(mean_vol = True, save_plots=True, save_mats=True), name = 'motion_correction')

    datasink = Node(DataSink(base_directory=result_dir, container=output_dir), name='datasink')

    preprocessing =  Workflow(base_dir = opj(result_dir, working_dir), name = "preprocessing")

    preprocessing.connect([(infosource_preproc, selectfiles_preproc, [('subject_id', 'subject_id'),
                                                                        ('run_id', 'run_id')]),
                            (selectfiles_preproc, motion_correction, [('func', 'in_file')]),
                            (motion_correction, datasink, [('par_file', 'preprocess.@parameters_file')])
                            ])

    return preprocessing

def get_preprocessing_2nd_step(exp_dir, result_dir, working_dir, output_dir, subject_list, run_list, fwhm):
    """
    Returns the 2nd part of the preprocessing workflow.

    Parameters: 
        - exp_dir: str, directory where raw data are stored
        - result_dir: str, directory where results will be stored
        - working_dir: str, name of the sub-directory for intermediate results
        - output_dir: str, name of the sub-directory for final results
        - subject_list: list of str, list of subject for which you want to do the preprocessing
        - run_list: list of str, list of runs for which you want to do the preprocessing 
        - fwhm: float, fwhm for smoothing step
        
    Returns: 
        - preprocessing: Nipype WorkFlow 
    """
    infosource_preproc = Node(IdentityInterface(fields = ['subject_id', 'run_id']), 
        name = 'infosource_preproc')

    infosource_preproc.iterables = [('subject_id', subject_list), ('run_id', run_list)]

    # Templates to select files node
    anat_file = opj('sub-{subject_id}', 'anat', 
                    'sub-{subject_id}_T1w.nii.gz')

    func_file = opj('sub-{subject_id}', 'func', 
                    'sub-{subject_id}_task-MGT_run-{run_id}_bold.nii.gz')
    
    mean_file = opj(result_dir, working_dir, 'preprocessing', '_run_id_{run_id}_subject_id_{subject_id}', 
                   'motion_correction', 'sub-{subject_id}_task-MGT_run-{run_id}_bold_mcf.nii.gz_mean_reg.nii.gz')
    
    motion_corrected_file = opj(result_dir, working_dir, 'preprocessing', '_run_id_{run_id}_subject_id_{subject_id}', 
                   'motion_correction', 'sub-{subject_id}_task-MGT_run-{run_id}_bold_mcf.nii.gz')
    
    parameters_file = opj(result_dir, working_dir, 'preprocessing', '_run_id_{run_id}_subject_id_{subject_id}', 
                   'motion_correction', 'sub-{subject_id}_task-MGT_run-{run_id}_bold_mcf.nii.gz.par')

    template = {'anat' : anat_file, 'func' : func_file, 'mean':mean_file, 'motion_corrected':motion_corrected_file,
               'param':parameters_file}

    # SelectFiles node - to select necessary files
    selectfiles_preproc = Node(SelectFiles(template, base_directory=exp_dir), name = 'selectfiles_preproc')

    skullstrip = Node(BET(frac = 0.3, robust = True), name = 'skullstrip')

    fast = Node(FAST(), name='fast')
    #register.connect(stripper, 'out_file', fast, 'in_files')

    binarize = Node(ImageMaths(op_string='-nan -thr 0.5 -bin'), name='binarize')
    pickindex = lambda x, i: x[i]
    #register.connect(fast, ('partial_volume_files', pickindex, 2), binarize,
    #                 'in_file')

    motion_correction = Node(MCFLIRT(mean_vol = True, save_plots=True, save_mats=True), name = 'motion_correction')

    mean2anat = Node(FLIRT(dof = 12), name = 'mean2anat')

    mean2anat_bbr = Node(FLIRT(dof = 12, cost = 'bbr', schedule = opj(os.getenv('FSLDIR'), 'etc/flirtsch/bbr.sch')), 
        name = 'mean2anat_bbr')

    anat2mni_linear = Node(FLIRT(dof = 12, reference = Info.standard_image('MNI152_T1_2mm.nii.gz')), 
        name = 'anat2mni_linear')

    anat2mni_nonlinear = Node(FNIRT(fieldcoeff_file = True, ref_file = Info.standard_image('MNI152_T1_2mm.nii.gz')), 
        name = 'anat2mni_nonlinear')

    warp_all = Node(ApplyWarp(interp='spline', ref_file = Info.standard_image('MNI152_T1_2mm.nii.gz')),
        name='warp_all')

    smooth = create_susan_smooth()
    smooth.inputs.inputnode.fwhm = fwhm
    smooth.inputs.inputnode.mask_file = Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz')

    datasink = Node(DataSink(base_directory=result_dir, container=output_dir), name='datasink')

    preprocessing =  Workflow(base_dir = opj(result_dir, working_dir), name = "preprocessing")

    preprocessing.connect([(infosource_preproc, selectfiles_preproc, [('subject_id', 'subject_id'),
                                                                        ('run_id', 'run_id')]),
                            (selectfiles_preproc, skullstrip, [('anat', 'in_file')]),
                            (selectfiles_preproc, motion_correction, [('func', 'in_file')]),
                            (skullstrip, fast, [('out_file', 'in_files')]),
                            (fast, binarize, [(('partial_volume_files', pickindex, 2), 'in_file')]),
                            (selectfiles_preproc, mean2anat, [('mean', 'in_file')]),
                            (skullstrip, mean2anat, [('out_file', 'reference')]), 
                            (selectfiles_preproc, mean2anat_bbr, [('mean', 'in_file')]),
                            (binarize, mean2anat_bbr, [('out_file', 'wm_seg')]),
                            (selectfiles_preproc, mean2anat_bbr, [('anat', 'reference')]),
                            (mean2anat, mean2anat_bbr, [('out_matrix_file', 'in_matrix_file')]),
                            (skullstrip, anat2mni_linear, [('out_file', 'in_file')]),
                            (anat2mni_linear, anat2mni_nonlinear, [('out_matrix_file', 'affine_file')]),
                            (selectfiles_preproc, anat2mni_nonlinear, [('anat', 'in_file')]),
                            (selectfiles_preproc, warp_all, [('motion_corrected', 'in_file')]),
                            (mean2anat_bbr, warp_all, [('out_matrix_file', 'premat')]),
                            (anat2mni_nonlinear, warp_all, [('fieldcoeff_file', 'field_file')]),
                            (warp_all, smooth, [('out_file', 'inputnode.in_files')]), 
                            (smooth, datasink, [('outputnode.smoothed_files', 'preprocess.@smoothed_file')]), 
                            (selectfiles_preproc, datasink, [('param', 'preprocess.@parameters_file')])
                            ])

    return preprocessing

def get_session_infos(event_file):
    '''
    Create Bunchs for specifyModel.
    
    Parameters :
    - event_file : str, file corresponding to the run and the subject to analyze
    
    Returns :
    - subject_info : list of Bunch for 1st level analysis.
    '''
    from os.path import join as opj
    from nipype.interfaces.base import Bunch
    
    cond_names = ['trial', 'gain', 'loss']
    
    onset = {}
    duration = {}
    amplitude = {}
    
    for c in cond_names:  # For each condition.
        onset.update({c : []}) # creates dictionary items with empty lists
        duration.update({c : []}) 
        amplitude.update({c : []})

    with open(event_file, 'rt') as f:
        next(f)  # skip the header

        for line in f:
            info = line.strip().split()
            # Creates list with onsets, duration and loss/gain for amplitude (FSL)
            for c in cond_names:
                if info[5] != 'NoResp':
                    onset[c].append(float(info[0]))
                    duration[c].append(float(4))
                    if c == 'gain':
                        amplitude[c].append(float(info[2]))
                    elif c == 'loss':
                        amplitude[c].append(float(info[3]))
                    elif c == 'trial':
                        amplitude[c].append(float(1))
                    elif c == 'response':
                        if info[5] == 'weakly_accept':
                            amplitude[c].append(float(1))
                        elif info[5] == 'strongly_accept':
                            amplitude[c].append(float(1))
                        elif info[5] == 'weakly_reject':
                            amplitude[c].append(float(0))
                        elif info[5] == 'strongly_reject':
                            amplitude[c].append(float(0))
                        else:
                            amplitude[c].append(float(0))
                        

    subject_info = []

    subject_info.append(Bunch(conditions=cond_names,
                             onsets=[onset[k] for k in cond_names],
                             durations=[duration[k] for k in cond_names],
                             amplitudes=[amplitude[k] for k in cond_names],
                             regressor_names=None,
                             regressors=None))

    return subject_info

# Linear contrast effects: 'Gain' vs. baseline, 'Loss' vs. baseline.
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
    # list of condition names     
    conditions = ['gain', 'loss']
    
    # create contrasts
    gain = ('gain', 'T', conditions, [1, 0])
    
    loss = ('loss', 'T', conditions, [0, 1])
    
    # contrast list
    contrasts = [gain, loss]

    return contrasts


def rm_preproc_files(files, subject_id, run_id, result_dir, working_dir):
    import shutil
    from os.path import join as opj

    preproc_dir = opj(result_dir, working_dir, 'preprocessing', f"_run_id_{run_id}_subject_id_{subject_id}")
    
    try:
        shutil.rmtree(preproc_dir)
    except OSError as e:
        print(e)
    else:
        print("The directory is deleted successfully")

def get_l1_analysis(subject_list, run_list, TR, exp_dir, output_dir, working_dir, result_dir):
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
	# Infosource Node - To iterate on subject and runs 
	infosource = Node(IdentityInterface(fields = ['subject_id', 'run_id']), name = 'infosource')
	infosource.iterables = [('subject_id', subject_list),
	                       ('run_id', run_list)]

	# Templates to select files node
	func_file = opj(result_dir, output_dir, 'preprocess', '_run_id_{run_id}_subject_id_{subject_id}','sub-{subject_id}_task-MGT_run-{run_id}_bold_mcf_warp_smooth.nii.gz')

	event_file = opj('sub-{subject_id}', 'func', 'sub-{subject_id}_task-MGT_run-{run_id}_events.tsv')
    
	param_file = opj(result_dir, output_dir, 'preprocess', '_run_id_{run_id}_subject_id_{subject_id}', 'sub-{subject_id}_task-MGT_run-{run_id}_bold_mcf.nii.gz.par')

	template = {'func' : func_file, 'event' : event_file, 'param' : param_file}

	# SelectFiles node - to select necessary files
	selectfiles = Node(SelectFiles(template, base_directory=exp_dir), name = 'selectfiles')

	# DataSink Node - store the wanted results in the wanted repository
	datasink = Node(DataSink(base_directory=result_dir, container=output_dir), name='datasink')
    
	# Node contrasts to get contrasts 
	contrasts = Node(Function(function=get_contrasts,
	                          input_names=['subject_id'],
	                          output_names=['contrasts']),
	                 name='contrasts')

	# Get Subject Info - get subject specific condition information
	get_subject_infos = Node(Function(input_names=['event_file'],
	                               output_names=['subject_info'],
	                               function=get_session_infos),
	                      name='get_subject_infos')

	specify_model = Node(SpecifyModel(high_pass_filter_cutoff = 60,
	                                 input_units = 'secs',
	                                 time_repetition = TR), name = 'specify_model')

	l1_design = Node(Level1Design(bases = {'dgamma':{'derivs' : True}},
	                             interscan_interval = TR, 
	                             model_serial_correlations = True), 
                     name = 'l1_design')

	model_generation = Node(FEATModel(), name = 'model_generation')

	model_estimate = Node(FILMGLS(), name='model_estimate')

	remove_preproc = Node(Function(input_names = ['subject_id', 'run_id', 'files', 'result_dir', 'working_dir'],
		function = rm_preproc_files), name = 'remove_preproc')

	remove_preproc.inputs.result_dir = result_dir
	remove_preproc.inputs.working_dir = working_dir

	# Create l1 analysis workflow and connect its nodes
	l1_analysis = Workflow(base_dir = opj(result_dir, working_dir), name = "l1_analysis")

	l1_analysis.connect([(infosource, selectfiles, [('subject_id', 'subject_id'),
	                                               ('run_id', 'run_id')]),
	                    (selectfiles, get_subject_infos, [('event', 'event_file')]),
	                    (infosource, contrasts, [('subject_id', 'subject_id')]),
                        (selectfiles, specify_model, [('param', 'realignment_parameters')]),
	                    (selectfiles, specify_model, [('func', 'functional_runs')]),
	                    (get_subject_infos, specify_model, [('subject_info', 'subject_info')]),
	                    (contrasts, l1_design, [('contrasts', 'contrasts')]),
	                    (specify_model, l1_design, [('session_info', 'session_info')]),
	                    (l1_design, model_generation, [('ev_files', 'ev_files'), ('fsf_files', 'fsf_file')]),
	                    (selectfiles, model_estimate, [('func', 'in_file')]),
	                    (model_generation, model_estimate, [('con_file', 'tcon_file'), 
	                                                        ('design_file', 'design_file')]),
	                    (model_estimate, datasink, [('results_dir', 'l1_analysis.@results')]),
	                    (model_generation, datasink, [('design_file', 'l1_analysis.@design_file'),
	                                                 ('design_image', 'l1_analysis.@design_img')])
	                    ])

	return l1_analysis

def get_l2_analysis(subject_list, contrast_list, run_list, exp_dir, output_dir, working_dir, result_dir):
	"""
	Returns the 2nd level of analysis workflow.

	Parameters: 
		- exp_dir: str, directory where raw data are stored
		- result_dir: str, directory where results will be stored
		- working_dir: str, name of the sub-directory for intermediate results
		- output_dir: str, name of the sub-directory for final results
		- subject_list: list of str, list of subject for which you want to do the preprocessing
		- contrast_list: list of str, list of contrasts to analyze
		- run_list: list of str, list of runs for which you want to do the analysis 


	Returns: 
	- l2_analysis: Nipype WorkFlow 
	"""      
	# Infosource Node - To iterate on subject and runs
	infosource_2ndlevel = Node(IdentityInterface(fields=['subject_id', 'contrast_id']), name='infosource_2ndlevel')
	infosource_2ndlevel.iterables = [('subject_id', subject_list), ('contrast_id', contrast_list)]

	# Templates to select files node
	copes_file = opj(output_dir, 'l1_analysis', '_run_id_*_subject_id_{subject_id}', 'results', 
	                 'cope{contrast_id}.nii.gz')

	varcopes_file = opj(output_dir, 'l1_analysis', '_run_id_*_subject_id_{subject_id}', 'results', 
	                    'varcope{contrast_id}.nii.gz')

	template = {'cope' : copes_file, 'varcope' : varcopes_file}

	# SelectFiles node - to select necessary files
	selectfiles_2ndlevel = Node(SelectFiles(template, base_directory=result_dir), name = 'selectfiles_2ndlevel')

	datasink_2ndlevel = Node(DataSink(base_directory=result_dir, container=output_dir), name='datasink_2ndlevel')

	# Generate design matrix
	specify_model_2ndlevel = Node(L2Model(num_copes = len(run_list)), name='l2model_2ndlevel')

	# Merge copes and varcopes files for each subject
	merge_copes_2ndlevel = Node(Merge(dimension='t'), name='merge_copes_2ndlevel')

	merge_varcopes_2ndlevel = Node(Merge(dimension='t'), name='merge_varcopes_2ndlevel')

	# Second level (single-subject, mean of all four scans) analyses: Fixed effects analysis.
	flame = Node(FLAMEO(run_mode = 'fe', mask_file = Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz')), 
	             name='flameo')

	l2_analysis = Workflow(base_dir = opj(result_dir, working_dir), name = "l2_analysis")

	l2_analysis.connect([(infosource_2ndlevel, selectfiles_2ndlevel, [('subject_id', 'subject_id'), 
	                                                                  ('contrast_id', 'contrast_id')]),
	                    (selectfiles_2ndlevel, merge_copes_2ndlevel, [('cope', 'in_files')]),
	                    (selectfiles_2ndlevel, merge_varcopes_2ndlevel, [('varcope', 'in_files')]),
	                    (merge_copes_2ndlevel, flame, [('merged_file', 'cope_file')]),
	                    (merge_varcopes_2ndlevel, flame, [('merged_file', 'var_cope_file')]),
	                    (specify_model_2ndlevel, flame, [('design_mat', 'design_file'), 
	                                                     ('design_con', 't_con_file'),
	                                                     ('design_grp', 'cov_split_file')]),
	                    (flame, datasink_2ndlevel, [('zstats', 'l2_analysis.@stats'),
	                                               ('tstats', 'l2_analysis.@tstats'),
	                                               ('copes', 'l2_analysis.@copes'),
	                                               ('var_copes', 'l2_analysis.@varcopes')])])

	return l2_analysis

def get_subgroups_contrasts(copes, varcopes, subject_ids, participants_file):
    ''' 
    Parameters :
    - copes: original file list selected by selectfiles node 
    - varcopes: original file list selected by selectfiles node 
    - subject_ids: list of subject IDs that are analyzed
    - participants_file: str, file containing participants characteristics
    
    This function return the file list containing only the files belonging to subject in the wanted group.
    '''
    
    from os.path import join as opj
    
    equalRange_id = []
    equalIndifference_id = []
    
    subject_list = ['sub-' + str(i) for i in subject_ids]
    
    with open(participants_file, 'rt') as f:
            next(f)  # skip the header
            
            for line in f:
                info = line.strip().split()
                
                if info[0] in subject_list and info[1] == "equalIndifference":
                    equalIndifference_id.append(info[0][-3:])
                elif info[0] in subject_list and info[1] == "equalRange":
                    equalRange_id.append(info[0][-3:])
                    
    copes_equalIndifference = []
    copes_equalRange = []
    varcopes_equalIndifference = []
    varcopes_equalRange = []
    
    for file in copes:
        sub_id = file.split('/')
        if sub_id[-2][-3:] in equalIndifference_id:
            copes_equalIndifference.append(file)
        elif sub_id[-2][-3:] in equalRange_id:
            copes_equalRange.append(file)     
            
    for file in varcopes:
        sub_id = file.split('/')
        if sub_id[-2][-3:] in equalIndifference_id:
            varcopes_equalIndifference.append(file)
        elif sub_id[-2][-3:] in equalRange_id:
            varcopes_equalRange.append(file) 
            
    return copes_equalIndifference, copes_equalRange, varcopes_equalIndifference, varcopes_equalRange, equalIndifference_id, equalRange_id 

def get_regs(equalRange_id, equalIndifference_id, method, subject_list):
	"""
	Create dictionary of regressors for group analysis. 

	Parameters: 
		- equalRange_id: list of str, ids of subjects in equal range group
		- equalIndifference_id: list of str, ids of subjects in equal indifference group
		- method: one of "equalRange", "equalIndifference" or "groupComp"

	Returns:
		- regressors: dict, dictionary of regressors used to distinguish groups in FSL group analysis
	"""
	if method == "equalRange":
		regressors = dict(group_mean = [1 for i in range(len(equalRange_id))])
        
	elif method == "equalIndifference":
		regressors = dict(group_mean = [1 for i in range(len(equalIndifference_id))])
        
	elif method == "groupComp":   
		equalRange_reg = [1 for i in range(len(equalRange_id) + len(equalIndifference_id))]
		equalIndifference_reg = [0 for i in range(len(equalRange_id) + len(equalIndifference_id))]
        
		for i, sub_id in enumerate(subject_list): 
			if sub_id in equalIndifference_id:
				index = i
				equalIndifference_reg[index] = 1
				equalRange_reg[index] = 0
            
		regressors = dict(equalRange = equalRange_reg, 
                      equalIndifference = equalIndifference_reg)
    
	return regressors

def get_group_workflow(subject_list, n_sub, contrast_list, method, exp_dir, output_dir, 
                       working_dir, result_dir):
	"""
	Returns the group level of analysis workflow.

	Parameters: 
		- exp_dir: str, directory where raw data are stored
		- result_dir: str, directory where results will be stored
		- working_dir: str, name of the sub-directory for intermediate results
		- output_dir: str, name of the sub-directory for final results
		- subject_list: list of str, list of subject for which you want to do the preprocessing
		- contrast_list: list of str, list of contrasts to analyze
		- method: one of "equalRange", "equalIndifference" or "groupComp"
		- n_sub: number of subject in subject_list
        
	Returns: 
		- l2_analysis: Nipype WorkFlow 
	"""     
	# Infosource Node - To iterate on subject and runs 
	infosource_3rdlevel = Node(IdentityInterface(fields = ['contrast_id', 'exp_dir', 'result_dir', 
                                                          'output_dir', 'working_dir', 'subject_list', 'method'],
                                                exp_dir = exp_dir, result_dir = result_dir, 
                                                 output_dir = output_dir, working_dir = working_dir, 
                                                 subject_list = subject_list, method = method), 
                               name = 'infosource_3rdlevel')
	infosource_3rdlevel.iterables = [('contrast_id', contrast_list)]

	# Templates to select files node
	copes_file = opj(output_dir, 'l2_analysis', '_contrast_id_{contrast_id}_subject_id_*', 
                     'cope1.nii.gz')
    
	varcopes_file = opj(output_dir, 'l2_analysis', '_contrast_id_{contrast_id}_subject_id_*', 
                     'varcope1.nii.gz')
    
	participants_file = opj(exp_dir, 'participants.tsv')

	template = {'cope' : copes_file, 'varcope' : varcopes_file, 'participants' : participants_file}

    # SelectFiles node - to select necessary files
	selectfiles_3rdlevel = Node(SelectFiles(template, base_directory=result_dir), name = 'selectfiles_3rdlevel')
    
	datasink_3rdlevel = Node(DataSink(base_directory=result_dir, container=output_dir), name='datasink_3rdlevel')
    
	merge_copes_3rdlevel = Node(Merge(dimension = 't'), name = 'merge_copes_3rdlevel')
	merge_varcopes_3rdlevel = Node(Merge(dimension = 't'), name = 'merge_varcopes_3rdlevel')
    
	subgroups_contrasts = Node(Function(input_names = ['copes', 'varcopes', 'subject_ids', 'participants_file'], 
                                  output_names = ['copes_equalIndifference', 'copes_equalRange',
                                                  'varcopes_equalIndifference', 'varcopes_equalRange', 
                                                 'equalIndifference_id', 'equalRange_id'],
                                  function = get_subgroups_contrasts), 
                         name = 'subgroups_contrasts')
    
	specifymodel_3rdlevel = Node(MultipleRegressDesign(), name = 'specifymodel_3rdlevel')

	flame_3rdlevel = Node(FLAMEO(run_mode = 'flame1', mask_file = Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz')), 
	             name='flame_3rdlevel')
    
	regs = Node(Function(input_names = ['equalRange_id', 'equalIndifference_id', 'method', 'subject_list'],
                                        output_names = ['regressors'],
                                        function = get_regs), name = 'regs')
	regs.inputs.method = method
	regs.inputs.subject_list = subject_list
    
	cluster = MapNode(Cluster(threshold = 3.1, out_threshold_file = True), name = 'cluster', 
                      iterfield = ['in_file', 'cope_file'], synchronize = True)
    
	l3_analysis = Workflow(base_dir = opj(result_dir, working_dir), name = f"l3_analysis_{method}_nsub_{n_sub}")
    
	l3_analysis.connect([(infosource_3rdlevel, selectfiles_3rdlevel, [('contrast_id', 'contrast_id')]),
                        (infosource_3rdlevel, subgroups_contrasts, [('subject_list', 'subject_ids')]),
                        (selectfiles_3rdlevel, subgroups_contrasts, [('cope', 'copes'), ('varcope', 'varcopes'),
                                                                    ('participants', 'participants_file')]),
                        (subgroups_contrasts, regs, [('equalRange_id', 'equalRange_id'),
                                                    ('equalIndifference_id', 'equalIndifference_id')]),
                        (regs, specifymodel_3rdlevel, [('regressors', 'regressors')])])


	if method == 'equalIndifference' or method == 'equalRange':
		specifymodel_3rdlevel.inputs.contrasts = [["group_mean", "T", ["group_mean"], [1]], ["group_mean_neg", "T", ["group_mean"], [-1]]]
        
		if method == 'equalIndifference':
			l3_analysis.connect([(subgroups_contrasts, merge_copes_3rdlevel, 
                                 [('copes_equalIndifference', 'in_files')]), 
                                (subgroups_contrasts, merge_varcopes_3rdlevel, 
                                 [('varcopes_equalIndifference', 'in_files')])])
		elif method == 'equalRange':
			l3_analysis.connect([(subgroups_contrasts, merge_copes_3rdlevel, [('copes_equalRange', 'in_files')]),
                                (subgroups_contrasts, merge_varcopes_3rdlevel, [('varcopes_equalRange', 'in_files')])])
            
	elif method == "groupComp":
		specifymodel_3rdlevel.inputs.contrasts = [["equalRange_sup", "T", ["equalRange", "equalIndifference"],
                                                   [1, -1]]]
        
		l3_analysis.connect([(selectfiles_3rdlevel, merge_copes_3rdlevel, [('cope', 'in_files')]),
                            (selectfiles_3rdlevel, merge_varcopes_3rdlevel, [('varcope', 'in_files')])])
    
	l3_analysis.connect([(merge_copes_3rdlevel, flame_3rdlevel, [('merged_file', 'cope_file')]),
                         (merge_varcopes_3rdlevel, flame_3rdlevel, [('merged_file', 'var_cope_file')]),
                        (specifymodel_3rdlevel, flame_3rdlevel, [('design_mat', 'design_file'),
                                                           ('design_con', 't_con_file'), 
                                                           ('design_grp', 'cov_split_file')]),
                        (flame_3rdlevel, cluster, [('zstats', 'in_file'), 
                                                  ('copes', 'cope_file')]),
                        (flame_3rdlevel, datasink_3rdlevel, [('zstats', 
                                                         f"l3_analysis_{method}_nsub_{n_sub}.@zstats"), 
                                                            ('tstats', 
                                                         f"l3_analysis_{method}_nsub_{n_sub}.@tstats")]), 
                        (cluster, datasink_3rdlevel, [('threshold_file', f"l3_analysis_{method}_nsub_{n_sub}.@thresh")])])
    
	return l3_analysis

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
    import nibabel as nib
    import numpy as np

    h1 = opj(result_dir, output_dir, f"l3_analysis_equalIndifference_nsub_{n_sub}", '_contrast_id_1')
    h2 = opj(result_dir, output_dir, f"l3_analysis_equalRange_nsub_{n_sub}", '_contrast_id_1')
    h3 = opj(result_dir, output_dir, f"l3_analysis_equalIndifference_nsub_{n_sub}", '_contrast_id_1')
    h4 = opj(result_dir, output_dir, f"l3_analysis_equalRange_nsub_{n_sub}", '_contrast_id_1')
    h5 = opj(result_dir, output_dir, f"l3_analysis_equalIndifference_nsub_{n_sub}", '_contrast_id_2')
    h6 = opj(result_dir, output_dir, f"l3_analysis_equalRange_nsub_{n_sub}", '_contrast_id_2')
    h7 = opj(result_dir, output_dir, f"l3_analysis_equalIndifference_nsub_{n_sub}", '_contrast_id_2')
    h8 = opj(result_dir, output_dir, f"l3_analysis_equalRange_nsub_{n_sub}", '_contrast_id_2')
    h9 = opj(result_dir, output_dir, f"l3_analysis_groupComp_nsub_{n_sub}", '_contrast_id_2')

    h = [h1, h2, h3, h4, h5, h6, h7, h8, h9]

    repro_unthresh = [opj(filename, "zstat2.nii.gz") if i in [4, 5] else opj(filename, "zstat1.nii.gz") for i, filename in enumerate(h)]

    repro_thresh = [opj(filename, '_cluster1', "zstat2_threshold.nii.gz") if i in [4, 5] else opj(filename, '_cluster0', "zstat1_threshold.nii.gz")  for i, filename in enumerate(h)]
    
    if not os.path.isdir(opj(result_dir, "NARPS-reproduction")):
        os.mkdir(opj(result_dir, "NARPS-reproduction"))
    
    for i, filename in enumerate(repro_unthresh):
        f_in = filename
        f_out = opj(result_dir, "NARPS-reproduction", f"team_{team_ID}_nsub_{n_sub}_hypo{i+1}_unthresholded.nii.gz")
        shutil.copyfile(f_in, f_out)

    for i, filename in enumerate(repro_thresh):
        f_in = filename 
        f_out = opj(result_dir, "NARPS-reproduction", f"team_{team_ID}_nsub_{n_sub}_hypo{i+1}_thresholded.nii.gz")
        shutil.copyfile(f_in, f_out)

    print(f"Results files of team {team_ID} reorganized.")
