
% first-level job ----

% $ narps_description -t UK24 -d preprocessing --json


clear matlabbatch;

% We consider that anat and func data have been gunzipped first

% % gunzip anat
% matlabbatch{1}.cfg_basicio.file_dir.file_ops.cfg_gunzip_files.files = {'/Users/camaumet/Softs/narps_open_pipelines/data/original/ds001734/sub-001/anat/sub-001_T1w.nii.gz'};
% matlabbatch{1}.cfg_basicio.file_dir.file_ops.cfg_gunzip_files.outdir = {''};
% matlabbatch{1}.cfg_basicio.file_dir.file_ops.cfg_gunzip_files.keep = true;
% 
% % gunzip sbref
% matlabbatch{end+1}.cfg_basicio.file_dir.file_ops.cfg_gunzip_files.files = {'/Users/camaumet/Softs/narps_open_pipelines/data/original/ds001734/sub-001/anat/sub-001_T1w.nii.gz'};
% matlabbatch{end}.cfg_basicio.file_dir.file_ops.cfg_gunzip_files.outdir = {''};
% matlabbatch{end}.cfg_basicio.file_dir.file_ops.cfg_gunzip_files.keep = true;

% 1 - Coregistration of the structural T1w image to the reference 
% functional image (defined as the single volume 'sbref' image acquired 
% before the first functional run) \n
matlabbatch{1}.spm.spatial.coreg.estimate.ref(1) = {...
    '/ABS_PATH/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-01_sbref.nii,1'}; %SBREF_FILE
matlabbatch{end}.spm.spatial.coreg.estimate.source(1) = {...
    '/ABS_PATH/data/original/ds001734/sub-001/anat/sub-001_T1w.nii,1'}; %ANAT_FILE

% 2 - Segmentation of the coregistered T1w image into grey matter, white 
% matter and cerebrospinal fluid tissue maps.\n

% note le fichier ANAT va garder le meme nom mais en pratique il y a une dependance sur la sortie du coreg
matlabbatch{end+1}.spm.spatial.preproc.channel.vols = {...
    '/ABS_PATH/data/original/ds001734/sub-001/anat/sub-001_T1w.nii,1'}; % this is the coregistered ANAT

% --> output ANAT_SEG_PARAM

% 3 - Reslicing of coregistered T1w image and segmentations to the same 
% voxel space of the reference (sbref) image

% Note Camille: I am not sure how to do this with the matlabbatch (imcalc
% should work but maybe there is a more direct option)

% spm_reslice{{ANAT, c1ANAT, c2ANAT, c3ANAT}, SBREF}

% 4 - Functional MRI volume realignment: first realigning all 4 'sbref' 
% images to the one acquired before the first run; followed by realignment 
% of all volumes in a particular run to the first image in that run. 
% This is done implictly with multi-session realignment in SPM12.
matlabbatch{end+1}.spm.spatial.realign.estwrite.data = {
    {
    %Note: I included only 3 volumes for the sub-001_task-MGT_run-01_bold
    %but in practice there should be as many as volumes in the 4d file
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-01_sbref.nii,1'
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-01_bold.nii,1'
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-01_bold.nii,2'
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-01_bold.nii,3'
    },
    {
    %Note: I included only 3 volumes for the sub-001_task-MGT_run-02_bold
    %but in practice there should be as many as volumes in the 4d file
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-02_sbref.nii,1'
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-02_bold.nii,1'
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-02_bold.nii,2'
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-02_bold.nii,3'
    },
    {
    %Note: I included only 3 volumes for the sub-001_task-MGT_run-03_bold
    %but in practice there should be as many as volumes in the 4d file
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-03_sbref.nii,1'
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-03_bold.nii,1'
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-03_bold.nii,2'
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-03_bold.nii,3'
    },
    {
    %Note: I included only 3 volumes for the sub-001_task-MGT_run-04_bold
    %but in practice there should be as many as volumes in the 4d file
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-04_sbref.nii,1'
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-04_bold.nii,1'
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-04_bold.nii,2'
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-04_bold.nii,3'
    }
                                                    }';

% 5 - Spatial smoothing of fMRI data.
% "spatial_smoothing": "Spatial smoothing was conducted on realigned 
% functional data using SPM12's standard smoothing function, which used a 
% fixed Gaussian smoothing kernel with FWHM of 4mm (twice the voxel size). 
% Smoothing was performed in the native functional space.",

matlabbatch{end+1}.spm.spatial.smooth.data = {
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/rsub-001_task-MGT_run-01_sbref.nii'
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/rsub-001_task-MGT_run-02_bold.nii'
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/rsub-001_task-MGT_run-03_bold.nii'
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/rsub-001_task-MGT_run-04_bold.nii'
};
matlabbatch{end}.spm.spatial.smooth.fwhm = [4 4 4];

% 6 - Calculation of several quality metrics based on preprocessed data, 
% including framewise displacement derived from motion correction 
% (realignment) parameters.
% The method was adapted from Power et al. (2012), 
% using 50mm as the assumed radius for translating rotation parameters 
% into displacements, assuming small angle approximations. A threshold of 
% 0.5 mm was set to define so-called 'motion outlier' volumes, which after 
% application to the FD timeseries yielded a binary 'scrubbing' regressor 
% per run.

% Note: here we can use FramewiseDisplacement from nipype that to compute
% the framewise displacement from the motion parameters that are available
% in the rp_ files generated after realignment.


% 7 - Generation of nuisance regressors for 
% 1st level analysis, including motion scrubbing regressor defined using 
% framewise displacement, and two tissue signal regressors (white matter 
% and cerebrospinal fluid) derived from averaging the voxel values found 
% within the segmentaed tissue maps.

% Scrubbing nuisance regressor: any functional volume that has a
% framewise displacement greater than 0.5mm is set to one in the scrubbing 
% nuisance covariate (all other volumes are set to zero)

% --> Save as reg_scrubbing.txt

% white matter signal nuisance: average value of voxels in th functional
% volume that are part of white matter (as computed using the segmentation,
% threshold>0.5?)

% --> Save as reg_wm.txt

% CSF signal nuisance: average value of voxels in th functional
% volume that are part of white matter (as computed using the segmentation,
% threshold>0.5?)

% --> Save as reg_csf.txt

% "independent_vars_first_level": "The GLM design matrix for the 1st level 
% analysis consited of 17 explicitly specified regressors per run. Of these
% regressors, the first 8 related to experimental conditions, whereas the 
% remaining 9 were nuisance regressors (6 head motion parameter regressors,
% 2 tissue signal regressors, and 1 scrubbing regressor), none of which 
% were entered as interactions. 
% For the experimental conditions, we are 
% interested in the parametric effect of gain and of loss as related to a 
% mixed gambles task. Trial data, indicating the amount of possible gain 
% and the amount of possible loss, as well as trial timing, trial durations
% and reaction times were supplied for each run per subject. From this data
% a block-task regressor was derived respectively for all trials where the 
% gamble could result in a possible gain, in a possible loss, and in a 
% no-loss-nor-gain situation (i.e. 3 block-task regressors). In each case 
% of the possible gain/loss regressors, 2 parametric modulation regressors 
% were added: 1 to model the parametric size of the possible gain/loss, and
% 1 to model the parametric size of the reaction time. In the case of the 
% no-loss-nor-gain block-task regressor a single extra parametric 
% modulation regressor was added to model the parametric size of the 
% reaction time. 
% All parametric modulation regressors were mean centered 
% before adding them to the design matrix for the GLM analysis, which 
% implicitly (in SPM12) orthogonalises all parametric modulators in a 
% series with reference to the first one. No time-modulation of conditions 
% was included.\n\nThe canonical HRF basis (default SPM12 option) was used 
% for convolution of the 8 conditions. SPM12 automatically also included a 
% constant regressor in the design matrix, and filtered the data and design 
% matrix using a discrete cosine basis set with a specified cutoff 
% frequency of 128 Hz (default). This accounted for slow drifts in the data
% , and no explicit drift regressors were included in the design matrix."

% Note: the covariates for tissue signal regressors and scrubbing regressor
% not implemented yet

% 8 regresseurs : gain, gain_value, gain_RT, loss, loss_value, loss_RT, no gain no loss, RT no gain no loss
% 9 nuisance : 6 head motion parameter regressors, 2 tissue signal regressors, and 1 scrubbing regressor

matlabbatch{end+1}.spm.stats.fmri_spec.dir = '<UNDEFINED>';
matlabbatch{end}.spm.stats.fmri_spec.timing.units = 'secs';
matlabbatch{end}.spm.stats.fmri_spec.timing.RT = '<UNDEFINED>';
matlabbatch{end}.spm.stats.fmri_spec.timing.fmri_t = 16;
matlabbatch{end}.spm.stats.fmri_spec.timing.fmri_t0 = 8;
matlabbatch{end}.spm.stats.fmri_spec.sess.scans = '<UNDEFINED>';
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(1).name = 'gain';
% Below we should include the 'onset' values from sub-001_task-MGT_run-01_events
% for which we have participant_response=accept (either weakly or strongly)
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(1).onset = '<UNDEFINED>';
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(1).duration = 4;
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(1).pmod(1).name = 'gain_value';
% Below we should include the 'gain' values from sub-001_task-MGT_run-01_events
% for which we have participant_response=accept (either weakly or strongly)
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(1).pmod(1).param = '<UNDEFINED>';
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(1).pmod(1).poly = 1;
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(1).pmod(2).name = 'gain_RT';
% Below we should include the 'RT' values from sub-001_task-MGT_run-01_events
% for which we have participant_response=accept (either weakly or strongly)
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(1).pmod(2).param = '<UNDEFINED>';
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(1).pmod(2).poly = 1;
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(1).orth = 1;
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(2).name = 'loss';
% Below we should include the 'onset' values from sub-001_task-MGT_run-01_events
% for which we have participant_response=accept (either weakly or strongly)
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(2).onset = '<UNDEFINED>';
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(2).duration = 2;
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(2).tmod = 0;
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(2).pmod(1).name = 'loss_value';
% Below we should include the 'loss' values from sub-001_task-MGT_run-01_events
% for which we have participant_response=accept (either weakly or strongly)
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(2).pmod(1).param = '<UNDEFINED>';
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(2).pmod(1).poly = 1;
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(2).pmod(2).name = 'loss_RT';
% Below we should include the 'RT' values from sub-001_task-MGT_run-01_events
% for which we have participant_response=accept (either weakly or strongly)
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(2).pmod(2).param = '<UNDEFINED>';
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(2).pmod(2).poly = 1;
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(2).orth = 1;
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(3).name = 'no_gain_no_loss';
% Below we should include the 'RT' values from sub-001_task-MGT_run-01_events
% for which we have participant_response=reject (either weakly or strongly)
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(3).onset = '<UNDEFINED>';
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(3).duration = 2;
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(3).tmod = 0;
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(3).pmod.name = 'reaction_time';
% Below we should include the 'RT' values from sub-001_task-MGT_run-01_events
% for which we have participant_response=reject (either weakly or strongly)
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(3).pmod.param = '<UNDEFINED>';
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(3).pmod.poly = 1;
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(3).orth = 1;
matlabbatch{end}.spm.stats.fmri_spec.sess.multi_reg = {
        'ABS_PATH/rp_MOTION_REG_FILE.txt'
        'ABS_PATH/reg_scrubbing.txt'
        'ABS_PATH/reg_wm.txt'
        'ABS_PATH/reg_csf.txt'
                                                     };


% "analysis.inference_contrast_effect": "For the experimental conditions, 
% we are interested in the parametric effect of gain and of loss as related
% to a mixed gambles task. Trial data, indicating the amount of possible 
% gain and the amount of possible loss, as well as trial timing, trial 
% durations and reaction times were supplied for each run per subject. From
% this data a block-task regressor was derived respectively for all trials 
% where the gamble could result in a possible gain, in a possible loss, and
% in a no-loss-nor-gain situation (i.e. 3 block-task regressors). In each 
% case of the possible gain/loss regressors, 2 parametric modulation 
% regressors were added: 1 to model the parametric size of the possible 
% gain/loss, and 1 to model the parametric size of the reaction time. In 
% the case of the no-loss-nor-gain block-task regressor a single extra 
% parametric modulation regressor was added to model the parametric size of
% the reaction time. We hypothesized that when looking at the parametric 
% effect of GAIN in the brain, we should be interested in the parametric 
% modulators related to the 'possible gain' condition, i.e. the parametric 
% size of the possible gain trials, and the parametric size of the reaction
% time for said trials. Similarly, when looking at the parametric effect of
% LOSS, we should be interested in the parametric modulators related to the
% 'possible loss' condition, i.e. the parametric size of the possible loss 
% trials, and the parametric size of the reaction time related to these 
% trials. We thus had two contrasts for the 1st level analysis: the GAIN 
% contrast which set the two GAIN parametric modulator regressors to 1 and 
% all other GLM regressors to 0; and the LOSS contrast which set the two 
% LOSS parametric modulator regressors to 1 and all other GLM regressors to
% 0. For the 2nd level group analysis, where the effect of interest was a 
% greater positive response to LOSS in one group vs another, our two-sample
% t-test was run by setting the contrast [-1 1], assuming the first group 
% was set as the equal indifference group and that the LOSS contrasts 
% resulting from the 1st level analysis of all subjects were fed into this 
% analysis.",

matlabbatch{end+1}.spm.stats.con.spmmat = '<UNDEFINED>';
matlabbatch{end}.spm.stats.con.consess{1}.tcon.name = 'gain_param';
matlabbatch{end}.spm.stats.con.consess{1}.tcon.weights = [0 1 1 0 0 0 0 0];
matlabbatch{end}.spm.stats.con.consess{2}.tcon.name = 'loss_param';
matlabbatch{end}.spm.stats.con.consess{2}.tcon.weights = [0 0 0 0 1 1 0 0];

% --> Creates con001.nii and con002.nii

% "preprocessing.inter_subject_reg": "Intersubject registration was 
% conducted on contrast maps that were output from 1st level statistical 
% analysis. SPM12's normalise (write) function was used to transform these 
% contrast maps from native subject space into 2mm isotropic MNI space 
% (using the MNI template supplied with SPM12). This function required a 
% forward transformation field, which was output from SPM12's unified 
% segmentation step as a field mapping between subject native space and MNI
% space (see segmentation explanantion above). The normalise (write) 
% process performed interpolation using a 4th degree B-spline. Further 
% SPM12 defaults for normalise (write) and the unified segmentation step 
% (including warping and regularisation parameters) were kept as is.",

matlabbatch{end}.spm.spatial.normalise.write.subj.resample = ...
    {'ABS_PATH/ANAT_SEG_PARAM'};
matlabbatch{end+1}.spm.spatial.normalise.write.subj.resample = {
    'ABS_PATH/con001.nii,1'
    'ABS_PATH/con002.nii.nii,1'
                                                            };
% --> Creates wcon001.nii and wcon002.nii
