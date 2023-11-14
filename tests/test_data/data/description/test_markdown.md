# NARPS team description : 9Q6R
## General
* `teamID` : 9Q6R
* `NV_collection_link` : https://neurovault.org/collections/4765/
* `results_comments` : Note: Amygdala wasn't recruited for hypothesis tests 7-9, but the extended salience network was recruited in all contrasts (e.g. aINS, ACC). Based on looking at the unthresholded maps, hypotheses 8 and 9 would've been confirmed at lower cluster thresholds (i.e. z≥2.3 rather than z≥3.1).
* `preregistered` : No
* `link_preregistration_form` : NA
* `regions_definition` : Harvard-Oxford probabilistic cortical and subcortical atlases (Frontal Median Cortex, L+R Amyg, and L+R Accum for vmPFC, amyg, and VS, respectively). Also used Neurosynth to generate a mask based on the search term "ventral striatum" (height threshold at z>12, and cluster-extent at > 400mm^3)
* `softwares` : FSL 5.0.11, MRIQC, FMRIPREP
* `general_comments` : NA
## Exclusions
* `n_participants` : 104
* `exclusions_details` : N=104 (54 eq_indiff, 50 eq_range). Excluded sub-018, sub-030, sub-088, and sub-100. High motion during function runs: All four participants had at least one run where > 50% of the TRs contained FD > 0.2mm. 18, 30, and 100 in particular were constant movers (all 4 runs > 50% TRS > 0.2 mm FD) 
## Preprocessing
* `used_fmriprep_data` : No
* `preprocessing_order` :  - MRIQC and FMRIPREP run on a local HPC
- FSL used for mass univariate analyses, avoiding re-registration using this approach: https://www.youtube.com/watch?time_continue=7&v=U3tG7JMEf7M
* `brain_extraction` : Freesurfer (i.e. part of fmriprep default pipeline)
* `segmentation` : Freesurfer
* `slice_time_correction` : Not performed
* `motion_correction` : Framewise displacement, and six standard motion regressors (x, y, z, rotx, rotx, and rotz) within subjects.; generated via MRIQC 
* `motion` : 6
* `gradient_distortion_correction` : NA
* `intra_subject_coreg` : bbregister, flirt, default FMRIPREP
* `distortion_correction` : Fieldmap-less distortion correction within fmriprep pipeline (--use-syn-sdc)
* `inter_subject_reg` : ANTs, multiscale nonlinear mutual-information default within FMRIPREP pipeline.
* `intensity_correction` : Default fMRIPREP INU correction
* `intensity_normalization` : Default fMRIPREP INU normalization
* `noise_removal` : None
* `volume_censoring` : None
* `spatial_smoothing` : 5mm FWHM
* `preprocessing_comments` : NA
## Analysis
* `data_submitted_to_model` : 453 total volumes, 104 participants (54 eq_indiff, 50 eq_range)
* `spatial_region_modeled` : Whole-Brain
* `independent_vars_first_level` : Event-related design predictors:
- Modeled duration = 4
- EVs  (3): Mean-centered Gain, Mean-Centered Loss, Events (constant)
Block design:
- baseline not explicitly modeled
HRF:
- FMRIB's Linear Optimal Basis Sets
Movement regressors:
- FD, six parameters (x, y, z, RotX, RotY, RotZ)
* `RT_modeling` : none
* `movement_modeling` : 1
* `independent_vars_higher_level` : EVs (2): eq_indiff, eq_range
Contrasts in the group-level design matrix:
1 --> mean (1, 1)
2 --> eq_indiff (1, 0)
3 --> eq_range (0, 1)
4 --> indiff_gr_range (1, -1)
5 --> range_gr_indiff (-1, 1)
* `model_type` : Mass Univariate
* `model_settings` : First model: individual runs; 
Second model: higher-level analysis on lower-level FEAT directories in a fixed effects model at the participant-level; 
Third model: higher-level analysis on 3D COPE images from *.feat directories within second model *.gfeat; FLAME 1 (FMRIB's Local Analysis of Mixed Effects), with a cluster threshold of z≥3.1
* `inference_contrast_effect` : First-Level A (Run-level; not listed: linear basis functions, FSL FLOBs):
Model EVs (3): gain, loss, event
- COPE1: Pos Gain (1, 0, 0)
- COPE4: Neg Gain  (-1, 0, 0)
- COPE7: Pos Loss (0, 1, 0)
- COPE10: Neg Loss (0, -1, 0)
- COPE13: Events (0, 0, 1)
Confound EVs (7): Framewise Displacement, x, y, z, RotX, RotY, RotZ. Generated in MRIQC.

First-Level B (Participant-level):
- All COPEs from the runs modeled in a high-level FEAT fixed effect model

Second-Level (Group-level):
- Separate high-level FLAME 1 models run on COPE1, COPE4, COPE7, and COPE10. Hypotheses 1-4 answered using the COPE1 model, Hypotheses 5-6 answered using the COPE10 model, and Hypotheses 7-9 answered using the COPE7 model.
Model EVs (2): eq_indiff, eq_range
- mean (1, 1)
- eq_indiff (1, 0)
- eq_range (0, 1)
- indiff_gr_range (1, -1)
- range_gr_indiff (-1, 1) 
* `search_region` : Whole brain
* `statistic_type` : Cluster size
* `pval_computation` : Standard parametric inference
* `multiple_testing_correction` : GRF_theory based FEW correction at z≥3.1 in FSL
* `comments_analysis` : NA
## Categorized for analysis
* `region_definition_vmpfc` : atlas HOA
* `region_definition_striatum` : atlas HOA, neurosynth
* `region_definition_amygdala` : atlas HOA
* `analysis_SW` : FSL
* `analysis_SW_with_version` : FSL 5.0.11
* `smoothing_coef` : 5
* `testing` : parametric
* `testing_thresh` : p<0.001
* `correction_method` : GRTFWE cluster
* `correction_thresh_` : p<0.05
## Derived
* `n_participants` : 104
* `excluded_participants` : 018, 030, 088, 100
* `func_fwhm` : 5
* `con_fwhm` : 
## Comments
* `excluded_from_narps_analysis` : No
* `exclusion_comment` : N/A
* `reproducibility` : 2
* `reproducibility_comment` : 
