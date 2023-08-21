from nipype.interfaces.spm import (Coregister, Smooth, OneSampleTTestDesign, EstimateModel, EstimateContrast, 
                                   Level1Design, TwoSampleTTestDesign, RealignUnwarp, 
                                   Normalize12, NewSegment, FieldMap)
from nipype.interfaces.fsl import ExtractROI
from nipype.interfaces.spm import Threshold 
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

    preproc_dir = opj(result_dir, working_dir, 'preprocessing', f"_run_id_{run_id}_subject_id_{subject_id}")
    
    dir_to_rm = ['coreg', 'norm_func', 'gunzip_magnitude', 'gunzip_phasediff', 'fieldmap']
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

    
    dir_to_rm = ['gunzip_func', 'gunzip_anat']
    for dirs in dir_to_rm:
        try:
            shutil.rmtree(opj(preproc_dir, dirs))
        except OSError as e:
            print(e)
        else:
            print(f"The directory {dirs} is deleted successfully")
    

    return files

def get_vox_dims(volume):
    ''' 
    Function that gives the voxel dimension of an image. 
    Not used here but if we use it, modify the connection to : 
    (?, normalize_func, [('?', 'apply_to_files'),
                                    (('?', get_vox_dims),
                                     'write_voxel_sizes')])
    '''
    import nibabel as nb
    if isinstance(volume, list):
        volume = volume[0]
    nii = nb.load(volume)
    hdr = nii.header
    voxdims = hdr.get_zooms()
    return [float(voxdims[0]), float(voxdims[1]), float(voxdims[2])]


def get_preprocessing(exp_dir, result_dir, working_dir, output_dir, subject_list, run_list, fwhm):
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
    
    motion_corrected_file = opj(result_dir, working_dir, 'preprocessing', '_run_id_{run_id}_subject_id_{subject_id}', 'motion_correction', 'usub-{subject_id}_task-MGT_run-{run_id}_bold.nii')
    motion_par_file = opj(result_dir, working_dir, 'preprocessing', '_run_id_{run_id}_subject_id_{subject_id}', 'motion_correction', 'rp_sub-{subject_id}_task-MGT_run-{run_id}_bold.txt')
    mean_img_file = opj(result_dir, working_dir, 'preprocessing', '_run_id_{run_id}_subject_id_{subject_id}', 'motion_correction', 'meanusub-{subject_id}_task-MGT_run-{run_id}_bold.nii')

    magnitude_file = opj('sub-{subject_id}', 'fmap', 'sub-{subject_id}_magnitude1.nii.gz')

    phasediff_file = opj('sub-{subject_id}', 'fmap', 'sub-{subject_id}_phasediff.nii.gz')

    template = {'anat' : anat_file, 'func' : func_file, 'magnitude' : magnitude_file, 'phasediff' : phasediff_file, 
               'realigned_unwarped_files' : motion_corrected_file, 'realignment_parameters' : motion_par_file, 
               'mean_image' : mean_img_file}

    # SelectFiles node - to select necessary files
    selectfiles_preproc = Node(SelectFiles(template, base_directory=exp_dir), name = 'selectfiles_preproc')

    # GUNZIP NODE : SPM do not use .nii.gz files
    gunzip_anat = Node(Gunzip(), name = 'gunzip_anat')

    gunzip_func = Node(Gunzip(), name = 'gunzip_func')

    gunzip_magnitude = Node(Gunzip(), name = 'gunzip_magnitude')

    gunzip_phasediff = Node(Gunzip(), name = 'gunzip_phasediff')
    
    extract_epi = Node(ExtractROI(t_min = 10, t_size=1, output_type = 'NIFTI'), name = 'extract_ROI')

    # Fieldmaps nodes
    # We used the "Calculate VDM" routine of the FieldMap tool in SPM12. 
    # We set the subject's phase (sub.*phasediff.nii) 
    # and magnitude (sub.*magnitude1.nii) images, the TE times to 4.92 and 7.38, 
    # the blip direction to -1, the total EPI readout time to 29.15, and field map to non-EPI. 
    # For each run, we selected image 10 to distortion correct and asked to match the VDM file 
    # to the EPI image and to write out the distortion-corrected EPI image. 
    # We set the structural image for comparison with the distortion-corrected EPI image and 
    # matched the first to the latter. 
    # For all other parameters we used the default values. We did not use Jacobian modulation.

    fieldmap = Node(FieldMap(blip_direction = -1, echo_times=(4.92, 7.38), 
                                total_readout_time = 29.15, matchanat=True,
                                matchvdm=True, writeunwarped = True, maskbrain = False, thresh = 0), name = 'fieldmap')

    ## Motion correction
    # We used the "Realign & Unwarp" routine in SPM12. We kept the default values for each parameter,
    # except for quality-speed trade-off that was increased to highest quality (i.e., 1), 
    # interpolation method that was set to 7th degree B-spline, and image registration that was 
    # set to be performed w.r.t. the mean. 
    motion_correction = Node(RealignUnwarp(interp = 7, register_to_mean = True, quality = 1, reslice_mask=False), 
                                name = 'motion_correction')
    ## Coregistration 
    # We used the "Coregister: Estimate" routine in SPM12. We set the mean functional image calculated in 
    # the motion correction step as reference image and the structural image as source image. 
    # We kept the default values for all other parameters. 
    coreg = Node(Coregister(jobtype = 'estimate', write_mask = False), name = 'coreg')

    ## Segmentation 
    # We performed segmentation on the structural image for each subject by using the "Segment" 
    # routine in SPM12, with default values for each parameter and using the template tissue 
    # probability maps (grey matter, white matter, CSF, bone, soft tissue, and air/background) 
    # in the tpm folder of SPM12. 
    # We saved a bias-corrected version of the image and both inverse and forward deformation field 
    # images.  
    tissue1 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 1), 1, (True,False), (True, False)]
    tissue2 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 2), 1, (True,False), (True, False)]
    tissue3 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 3), 2, (True,False), (True, False)]
    tissue4 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 4), 3, (True,False), (True, False)]
    tissue5 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 5), 4, (True,False), (True, False)]
    tissue6 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 6), 2, (True,False), (True, False)]
    tissue_list = [tissue1, tissue2, tissue3, tissue4, tissue5, tissue6]
    
    seg = Node(NewSegment(write_deformation_fields = [True, True], tissues = tissue_list), name = 'seg')
    
    ## Normalization
    # We used the "Normalise: Write" routine in SPM12. We set the motion-corrected EPI images 
    # for each run as images to resample and the spatial normalization deformation field file 
    # obtained with the "Segment" routine as deformation field for the normalization procedure. 
    # We used default values for the bounding box and set voxel size to 2 x 2 x 2.4 mm and
    # interpolation method to 7th degree B-spline.
    norm_func = Node(Normalize12(jobtype = 'write', write_voxel_sizes = [2, 2, 2.4],
                                write_interp = 7), name = 'norm_func')

    ## Smoothing
    # We used the "Smooth" routine in SPM12. We selected the normalized EPI images and set 
    # the FWHM of the Gaussian smoothing kernel to 6mm. We used the default values for 
    # the other parameters.
    smooth = Node(Smooth(fwhm = 6, implicit_masking = False), name = 'smooth')

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
                           (selectfiles_preproc, gunzip_func, [('func', 'in_file')]),
                           (selectfiles_preproc, gunzip_anat, [('anat', 'in_file')]),
                           (selectfiles_preproc, gunzip_phasediff, [('phasediff', 'in_file')]),
                           (selectfiles_preproc, gunzip_magnitude, [('magnitude', 'in_file')]),
                           (gunzip_anat, fieldmap, [('out_file', 'anat_file')]),
                           (gunzip_func, extract_epi, [('out_file', 'in_file')]),
                           (extract_epi, fieldmap, [('roi_file', 'epi_file')]), 
                           (gunzip_magnitude, fieldmap, [('out_file', 'magnitude_file')]),
                           (gunzip_phasediff, fieldmap, [('out_file', 'phase_file')]),
                           (gunzip_func, motion_correction, [('out_file', 'in_files')]),
                           (fieldmap, motion_correction, [('vdm', 'phase_map')]),
                           (motion_correction, coreg, [('mean_image', 'target'), 
                                                       ('realigned_unwarped_files', 'apply_to_files')]),
                           (gunzip_anat, coreg, [('out_file', 'source')]),
                           (rm_gunzip, seg, [('files', 'channel_files')]),
                           (coreg, seg, [('coregistered_source', 'channel_files')]),
                           (coreg, norm_func, [('coregistered_files', 'apply_to_files')]),
                           (seg, norm_func, [('forward_deformation_field', 'deformation_file')]),
                           (norm_func, smooth, [('normalized_files', 'in_files')]),
                           (motion_correction, datasink_preproc, 
                            [('realignment_parameters', 'preprocess.@parameters')]),
                           (smooth, rm_files, [('smoothed_files', 'files')]),
                           (rm_files, datasink_preproc, [('files', 'preprocess.@smooth')]),
                           (seg, datasink_preproc, [('native_class_images', 'preprocess.@seg_maps_native'),
                                                    ('normalized_class_images', 'preprocess.@seg_maps_norm')])])
    
    return preprocessing    

def get_subject_infos(event_files, runs):
    '''
    The model contained 6 regressors per run:
    - One predictor with onset at the start of the trial and duration of 4s.
    - Two parametric modulators (one for gains, one for losses) were added to the trial onset predictor. 
    The two parametric modulators were orthogonalized w.r.t. the main predictor, but were not orthogonalized w.r.t. one another.
    - Two predictors modelling the decision output, one for accepting the gamble and one for rejecting it 
    (merging strong and weak decisions). 
    The onset was defined as the beginning of the trial + RT and the duration was set to 0 (stick function).
    - One constant term for each run was included (SPM12 default design).
    
    Create Bunchs for specifySPMModel.

    Parameters :
    - event_files: list of str, list of events files (one per run) for the subject
    - runs: list of str, list of runs to use
    
    Returns :
    - subject_info : list of Bunch for 1st level analysis.
    '''
    from nipype.interfaces.base import Bunch
    
    cond_names = ['trial', 'accepting', 'rejecting']
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
    
    for r, run in enumerate(runs):
        
        f_events = event_files[r]
        
        with open(f_events, 'rt') as f:
            next(f)  # skip the header
            
            for line in f:
                info = line.strip().split()
                
                for cond in cond_names:
                    val = cond + '_run' + str(r+1) # trial_run1 or accepting_run1
                    val_gain = 'gain_run' + str(r+1) # gain_run1
                    val_loss = 'loss_run' + str(r+1) # loss_run1
                    if cond == 'trial':
                        onset[val].append(float(info[0])) # onsets for trial_run1 
                        duration[val].append(float(4))
                        weights_gain[val_gain].append(float(info[2])) # weights gain for trial_run1
                        weights_loss[val_loss].append(float(info[3])) # weights loss for trial_run1
                    elif cond == 'accepting' and 'accept' in info[5]:
                        onset[val].append(float(info[0]) + float(info[4]))
                        duration[val].append(float(0))
                    elif cond == 'rejecting' and 'reject' in info[5]:
                        onset[val].append(float(info[0]) + float(info[4]))
                        duration[val].append(float(0))
                    

    # Bunching is done per run, i.e. trial_run1, trial_run2, etc.
    # But names must not have '_run1' etc because we concatenate runs 
    subject_info = []
    for r in range(len(runs)):

        cond = [s + '_run' + str(r+1) for s in cond_names]
        gain = 'gain_run' + str(r+1)
        loss = 'loss_run' + str(r+1)

        subject_info.insert(r,
                           Bunch(conditions=cond_names,
                                 onsets=[onset[c] for c in cond],
                                 durations=[duration[c] for c in cond],
                                 amplitudes=None,
                                 tmod=None,
                                 pmod=[Bunch(name=['gain', 'loss'],
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
    # list of condition names     
    conditions = ['trial', 'trialxgain^1', 'trialxloss^1']
    
    # create contrasts
    trial = ('trial', 'T', conditions, [1, 0, 0])
    
    effect_gain = ('effect_of_gain', 'T', conditions, [0, 1, 0])
    
    effect_loss = ('effect_of_loss', 'T', conditions, [0, 0, 1])
    
    # contrast list
    contrasts = [effect_gain, effect_loss]

    return contrasts


def compute_mask(wc1_file, wc2_file, wc3_file, result_dir, output_dir, subject_id):
    '''
    Function to compute mask to which we restrict the l1 analysis.
    
    Parameters:
        - wc1_file: str, file wc1 obtained using segmentation function from SPM
        - wc2_file: str, file wc2 obtained using segmentation function from SPM
        - wc3_file: str, file wc3 obtained using segmentation function from SPM
        - result_dir: str, directory where results will be stored
        - output_dir: str, name of the sub-directory for final results
        - subject_id: str, name of the subject for which we do the analysis
    '''

    from nilearn import masking, plotting
    import nibabel as nib
    from os.path import join as opj
    import os

    masks_list = []
    for mask in [wc1_file, wc2_file, wc3_file]:
        mask_img = nib.load(mask)
        mask_data = mask_img.get_fdata()
        mask_affine = mask_img.affine
        masks_list.append(mask_data)

    mask_img = masks_list[0] + masks_list[1] + masks_list[2] > 0.3
    mask_img = mask_img.astype('float64')
    mask = nib.Nifti1Image(mask_img, mask_affine)
    if not os.path.isdir(opj(result_dir, output_dir, 'l1_analysis')):
        os.mkdir(opj(result_dir, output_dir, 'l1_analysis'))
    if not os.path.isdir(opj(result_dir, output_dir, 'l1_analysis', f"_subject_id_{subject_id}")):
        os.mkdir(opj(result_dir, output_dir, 'l1_analysis', f"_subject_id_{subject_id}"))
    mask_path = opj(result_dir, output_dir, 'l1_analysis', f"_subject_id_{subject_id}", 'computed_mask.nii')
    nib.save(mask, mask_path)
    
    return mask_path


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
    func_file = opj(result_dir, output_dir, 'preprocess', '_run_id_*_subject_id_{subject_id}', 
                   'swusub-{subject_id}_task-MGT_run-*_bold.nii')

    event_files = opj(exp_dir, 'sub-{subject_id}', 'func', 
                      'sub-{subject_id}_task-MGT_run-*_events.tsv')
    
    wc1_file = opj(result_dir, output_dir, 'preprocess', '_run_id_01_subject_id_{subject_id}', 
                   'wc1sub-{subject_id}_T1w.nii')
    wc2_file = opj(result_dir, output_dir, 'preprocess', '_run_id_01_subject_id_{subject_id}', 
                   'wc2sub-{subject_id}_T1w.nii')
    wc3_file = opj(result_dir, output_dir, 'preprocess', '_run_id_01_subject_id_{subject_id}', 
                   'wc3sub-{subject_id}_T1w.nii')

    template = {'func' : func_file, 'event' : event_files, 'wc1' : wc1_file, 'wc2' : wc2_file, 'wc3' : wc3_file}

    # SelectFiles node - to select necessary files
    selectfiles = Node(SelectFiles(template, base_directory=exp_dir), name = 'selectfiles')
    
    # DataSink Node - store the wanted results in the wanted repository
    datasink = Node(DataSink(base_directory=result_dir, container=output_dir), name='datasink')

    # Get Subject Info - get subject specific condition information
    subject_infos = Node(Function(input_names=['event_files', 'runs'],
                                   output_names=['subject_info'],
                                   function=get_subject_infos),
                          name='subject_infos')
    
    subject_infos.inputs.runs = run_list
    
    # SpecifyModel - Generates SPM-specific Model
    specify_model = Node(SpecifySPMModel(concatenate_runs = True, input_units = 'secs', output_units = 'secs',
                                        time_repetition = TR, high_pass_filter_cutoff = 128), name='specify_model')
    
    # Compute mask from wc1,wc2,wc3 files
    mask = Node(Function(input_names = ['wc1_file', 'wc2_file', 'wc3_file', 'result_dir', 'output_dir', 'subject_id'],
                         output_names = ['mask_path'],
                         function = compute_mask), name = 'mask')
    
    mask.inputs.result_dir = result_dir
    mask.inputs.output_dir = output_dir
    

    # Level1Design - Generates an SPM design matrix
    l1_design = Node(Level1Design(bases = {'hrf': {'derivs': [0, 0]}}, timing_units = 'secs', 
                                  interscan_interval = TR, model_serial_correlations = 'AR(1)'),
                     name='l1_design')
    
    #l1_design.inputs.mask_image = 

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
                        (infosource, contrasts, [('subject_id', 'subject_id')]),
                        (infosource, mask, [('subject_id', 'subject_id')]),
                        (selectfiles, mask, [('wc1', 'wc1_file'), ('wc2', 'wc2_file'), ('wc3', 'wc3_file')]),
                        (mask, l1_design, [('mask_path', 'mask_image')]),
                        (subject_infos, specify_model, [('subject_info', 'subject_info')]),
                        (contrasts, contrast_estimate, [('contrasts', 'contrasts')]),
                        (selectfiles, subject_infos, [('event', 'event_files')]),
                        (selectfiles, specify_model, [('func', 'functional_runs')]), 
                        (specify_model, l1_design, [('session_info', 'session_info')]),
                        (l1_design, l1_estimate, [('spm_mat_file', 'spm_mat_file')]),
                        (l1_estimate, contrast_estimate, [('spm_mat_file', 'spm_mat_file'),
                                                          ('beta_images', 'beta_images'),
                                                          ('residual_image', 'residual_image')]),
                        (l1_estimate, datasink, [('mask_image', 'l1_analysis.@mask')]),
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
        - n_sub: float, number of subjects used to do the analysis
        - method: one of "equalRange", "equalIndifference" or "groupComp"

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
    threshold = MapNode(Threshold(use_fwe_correction = False, height_threshold = 0.001), name = "threshold", iterfield = ["stat_image", "contrast_index"])

    l2_analysis = Workflow(base_dir = opj(result_dir, working_dir), name = f"l2_analysis_{method}_nsub_{n_sub}")

    l2_analysis.connect([(infosource_groupanalysis, selectfiles_groupanalysis, [('contrast_id', 'contrast_id')]),
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

    elif method == 'groupComp':
        contrasts = [('Eq range vs Eq indiff in loss', 'T', ['Group_{1}', 'Group_{2}'], [1, -1])]
        # Node for the design matrix
        two_sample_t_test_design = Node(TwoSampleTTestDesign(), name = 'two_sample_t_test_design')

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
        - n_sub: float, number of subject used for the analysis
        - team_ID: str, ID of the team to reorganize results

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


