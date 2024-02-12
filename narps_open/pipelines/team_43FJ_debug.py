# pylint: skip-file
from nipype.interfaces.fsl import (BET, FAST, MCFLIRT, FLIRT, FNIRT, ApplyWarp, SUSAN, MotionOutliers,
                                   Info, ImageMaths, IsotropicSmooth, Threshold, Level1Design, FEATModel, 
                                   L2Model, Merge, FLAMEO, ContrastMgr, FILMGLS, Randomise, MultipleRegressDesign)
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.algorithms.misc import Gunzip
from nipype import Workflow, Node, MapNode
from nipype.interfaces.base import Bunch

from os.path import join as opj
import os

"""
Preprocessing was conducted using FEAT (FMRI Expert Analysis Tool) v 6.00, part of FSL version 5.0.10 (FMRIB Software Library, www.fmrib.ox.ac.uk/fsl). 
- Preprocessing consisted of nonbrain removal using BET (Brain Extraction Tool for FSL), 
- high-pass filtering (100-s cutoff), 
- and spatial smoothing using a Gaussian kernel of FWHM 5 mm. 
- Motion correction was performed with MCFLIRT (intra-modal motion correction tool based on optimization and registration techniques in FSL’s registration tool FLIRT) using 24 standard and extended regressors (six motion parameters, the derivatives of those parameters, and the squares of the derivatives and the original parameters). 
Additional spike regressors created using fsl_motion_outliers (frame displacement threshold=75th percentile plus 1.5 times the interquartile range) were also included. 
- Each participant’s functional data were registered to their T1 weighted anatomical image using boundary based registration (BBR; Greve & Fischl, 2009) and then to MNI (Montreal Neurological Institute) stereotaxic space with 12 degrees of freedom via FLIRT (FMRIB’s Linear Image Registration Tool). 
Alignment was visually confirmed for all participants. FILM (FMRIB’s Improved Linear Model) prewhitening was performed to estimate voxelwise autocorrelation and improve estimation efficiency.
"""

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

    template = {'anat' : anat_file, 'func' : func_file}

    # SelectFiles node - to select necessary files
    selectfiles_preproc = Node(SelectFiles(template, base_directory=exp_dir), name = 'selectfiles_preproc')
    
    """
    Nonbrain removal was performed using BET (Brain Extraction Tool for FSL). 
    Default fractional intensity threshold of .5 and vertical gradient of 0 were used. 
    """

    skullstrip = Node(BET(frac = 0.5, robust = True, vertical_gradient = 0), name = 'skullstrip')

    fast = Node(FAST(), name='fast')
    #register.connect(stripper, 'out_file', fast, 'in_files')

    binarize = Node(ImageMaths(op_string='-nan -thr 0.5 -bin'), name='binarize')
    pickindex = lambda x, i: x[i]
    #register.connect(fast, ('partial_volume_files', pickindex, 2), binarize,
    #                 'in_file')
    
    """
    Motion correction was performed with MCFLIRT (intra-modal motion correction tool based on optimization and registration techniques in FSL’s registration tool FLIRT) using 24 standard and extended regressors (six motion parameters, the derivatives of those parameters, and the squares of the derivatives and the original parameters). 
    Additional spike regressors created using fsl_motion_outliers (frame displacement threshold=75th percentile plus 1.5 times the interquartile range) were also included. 
    Middle volume default was used as the reference image. 
    All optimizations use trilinear interpolation. B0 and fieldmap unwarping were not used. 
    """

    motion_correction = Node(MCFLIRT(mean_vol = True, save_rms = True, save_plots=True, save_mats=True), name = 'motion_correction')
    
    motion_outliers = Node(MotionOutliers(), name = 'motion_outliers')
    
    """
    Each participant’s functional data were registered via linear interpolation to their T1 weighted anatomical image using boundary based registration (BBR; Greve & Fischl, 2009) and then to MNI (Montreal Neurological Institute) stereotaxic space with 12 degrees of freedom via FLIRT (FMRIB’s Linear Image Registration Tool).
    """

    mean2anat = Node(FLIRT(dof = 12), name = 'mean2anat')

    mean2anat_bbr = Node(FLIRT(dof = 12, cost = 'bbr', schedule = opj(os.getenv('FSLDIR'), 'etc/flirtsch/bbr.sch')), 
        name = 'mean2anat_bbr')

    anat2mni_linear = Node(FLIRT(dof = 12, reference = Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz')), 
        name = 'anat2mni_linear')

    anat2mni_nonlinear = Node(FNIRT(fieldcoeff_file = True, ref_file = Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz')), 
        name = 'anat2mni_nonlinear')

    warp_all = Node(ApplyWarp(interp='spline', ref_file = Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz')),
        name='warp_all')
    
    """
    Feat FWHM 5mm applied to each volume of functional data. 
    """

    smooth = Node(SUSAN(brightness_threshold = 2000, fwhm = 5), name = 'smooth')

    datasink = Node(DataSink(base_directory=result_dir, container=output_dir), name='datasink')

    preprocessing =  Workflow(base_dir = opj(result_dir, working_dir), name = "preprocessing")

    preprocessing.connect([(infosource_preproc, selectfiles_preproc, [('subject_id', 'subject_id'),
                                                                        ('run_id', 'run_id')]),
                            (selectfiles_preproc, skullstrip, [('anat', 'in_file')]),
                            (selectfiles_preproc, motion_correction, [('func', 'in_file')]),
                            (selectfiles_preproc, motion_outliers, [('func', 'in_file')]),
                            (skullstrip, fast, [('out_file', 'in_files')]),
                            (fast, binarize, [(('partial_volume_files', pickindex, 2), 'in_file')]),
                            (motion_correction, mean2anat, [('mean_img', 'in_file')]),
                            (skullstrip, mean2anat, [('out_file', 'reference')]), 
                            (motion_correction, mean2anat_bbr, [('mean_img', 'in_file')]),
                            (binarize, mean2anat_bbr, [('out_file', 'wm_seg')]),
                            (selectfiles_preproc, mean2anat_bbr, [('anat', 'reference')]),
                            (mean2anat, mean2anat_bbr, [('out_matrix_file', 'in_matrix_file')]),
                            (skullstrip, anat2mni_linear, [('out_file', 'in_file')]),
                            (anat2mni_linear, anat2mni_nonlinear, [('out_matrix_file', 'affine_file')]),
                            (selectfiles_preproc, anat2mni_nonlinear, [('anat', 'in_file')]),
                            (motion_correction, warp_all, [('out_file', 'in_file')]),
                            (mean2anat_bbr, warp_all, [('out_matrix_file', 'premat')]),
                            (anat2mni_nonlinear, warp_all, [('fieldcoeff_file', 'field_file')]),
                            (warp_all, smooth, [('out_file', 'in_file')]), 
                            (smooth, datasink, [('smoothed_file', 'preprocess.@smoothed_file')]), 
                            (motion_correction, datasink, [('par_file', 'preprocess.@parameters_file')]),
                            (motion_outliers, datasink, [('out_metric_values', 'preprocess.@outliers'), ('out_metric_plot', 'preprocess.@outliers_plot')]), 
                            ])

    return preprocessing


