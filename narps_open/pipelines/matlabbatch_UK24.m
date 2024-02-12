
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


% 7 - Generation of nuisance regressors for 
% 1st level analysis, including motion scrubbing regressor defined using 
% framewise displacement, and two tissue signal regressors (white matter 
% and cerebrospinal fluid) derived from averaging the voxel values found 
% within the segmentaed tissue maps.

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

% 8 regresseurs : gain, loss, no gain no loss, parametric gain, RT gain, parametric loss, RT loss, RT no gain no loss
% gain: onset of gambles that led to gain, duration=

matlabbatch{end+1}.spm.stats.fmri_spec.dir = '<UNDEFINED>';
matlabbatch{end}.spm.stats.fmri_spec.timing.units = 'secs';
matlabbatch{end}.spm.stats.fmri_spec.timing.RT = '<UNDEFINED>';
matlabbatch{end}.spm.stats.fmri_spec.timing.fmri_t = 16;
matlabbatch{end}.spm.stats.fmri_spec.timing.fmri_t0 = 8;
matlabbatch{end}.spm.stats.fmri_spec.sess.scans = '<UNDEFINED>';
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(1).name = 'gain';
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(1).onset = '<UNDEFINED>';
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(1).duration = 2;
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(1).tmod = 0;
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(1).pmod(1).name = 'money';
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(1).pmod(1).param = '<UNDEFINED>';
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(1).pmod(1).poly = 1;
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(1).pmod(2).name = 'reaction_time';
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(1).pmod(2).param = '<UNDEFINED>';
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(1).pmod(2).poly = 1;
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(1).orth = 1;
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(2).name = 'loss';
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(2).onset = '<UNDEFINED>';
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(2).duration = 2;
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(2).tmod = 0;
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(2).pmod(1).name = 'money';
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(2).pmod(1).param = '<UNDEFINED>';
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(2).pmod(1).poly = 1;
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(2).pmod(2).name = 'reaction_time';
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(2).pmod(2).param = '<UNDEFINED>';
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(2).pmod(2).poly = 1;
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(2).orth = 1;
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(3).name = 'no_gain_no_loss';
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(3).onset = '<UNDEFINED>';
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(3).duration = 2;
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(3).tmod = 0;
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(3).pmod.name = 'reaction_time';
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(3).pmod.param = '<UNDEFINED>';
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(3).pmod.poly = 1;
matlabbatch{end}.spm.stats.fmri_spec.sess.cond(3).orth = 1;
matlabbatch{end}.spm.stats.fmri_spec.sess.multi = {''};
matlabbatch{end}.spm.stats.fmri_spec.sess.regress = struct('name', {}, 'val', {});
matlabbatch{end}.spm.stats.fmri_spec.sess.multi_reg = {''};
matlabbatch{end}.spm.stats.fmri_spec.sess.hpf = 128;
matlabbatch{end}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
matlabbatch{end}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
matlabbatch{end}.spm.stats.fmri_spec.volt = 1;
matlabbatch{end}.spm.stats.fmri_spec.global = 'None';
matlabbatch{end}.spm.stats.fmri_spec.mthresh = 0.8;
matlabbatch{end}.spm.stats.fmri_spec.mask = {''};
matlabbatch{end}.spm.stats.fmri_spec.cvi = 'AR(1)';



% 
% % TODO: ignored for now
% 
% % ---------
% 
% 
% % --------
% 
% matlabbatch{end+1}.spm.util.imcalc.input(1) = {ANAT_FILE} 
% matlabbatch{end+1}.spm.util.imcalc.input(2) = cfg_dep('Segment: c1 Images', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{1}, '.','c', '()',{':'}));
% matlabbatch{end+1}.spm.util.imcalc.input(3) = cfg_dep('Segment: c2 Images', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{2}, '.','c', '()',{':'}));
% matlabbatch{end+1}.spm.util.imcalc.input(4) = cfg_dep('Segment: c3 Images', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{3}, '.','c', '()',{':'}));
% matlabbatch{end+1}.spm.util.imcalc.output = BRAIN_EXTRACTED;
% matlabbatch{end+1}.spm.util.imcalc.outdir = {PREPROC_DIR};
% matlabbatch{end+1}.spm.util.imcalc.expression = 'i1.*((i2+i3+i4)>0.5)';
% matlabbatch{end+1}.spm.util.imcalc.var = struct('name', {}, 'value', {});
% matlabbatch{end+1}.spm.util.imcalc.options.dmtx = 0;
% matlabbatch{end+1}.spm.util.imcalc.options.mask = 0;
% matlabbatch{end+1}.spm.util.imcalc.options.interp = 1;
% matlabbatch{end+1}.spm.util.imcalc.options.dtype = 4;
% 
% 
% % next !!
% 
% 
% 
% matlabbatch{5}.spm.spatial.normalise.write.subj(1).def(1) = cfg_dep('Segment: Forward Deformations', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','fordef', '()',{':'}));
% matlabbatch{5}.spm.spatial.normalise.write.subj(1).resample(1) = cfg_dep('Realign: Estimate & Reslice: Realigned Images (Sess 1)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{1}, '.','cfiles'));
% matlabbatch{5}.spm.spatial.normalise.write.subj(2).def(1) = cfg_dep('Segment: Forward Deformations', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','fordef', '()',{':'}));
% matlabbatch{5}.spm.spatial.normalise.write.subj(2).resample(1) = cfg_dep('Realign: Estimate & Reslice: Realigned Images (Sess 2)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{2}, '.','cfiles'));
% matlabbatch{5}.spm.spatial.normalise.write.subj(3).def(1) = cfg_dep('Segment: Forward Deformations', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','fordef', '()',{':'}));
% matlabbatch{5}.spm.spatial.normalise.write.subj(3).resample(1) = cfg_dep('Realign: Estimate & Reslice: Realigned Images (Sess 3)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{3}, '.','cfiles'));
% matlabbatch{5}.spm.spatial.normalise.write.subj(4).def(1) = cfg_dep('Segment: Forward Deformations', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','fordef', '()',{':'}));
% matlabbatch{5}.spm.spatial.normalise.write.subj(4).resample(1) = cfg_dep('Segment: Bias Corrected (1)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','channel', '()',{1}, '.','biascorr', '()',{':'}));
% matlabbatch{5}.spm.spatial.normalise.write.subj(5).def(1) = cfg_dep('Segment: Forward Deformations', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','fordef', '()',{':'}));
% matlabbatch{5}.spm.spatial.normalise.write.subj(5).resample(1) = cfg_dep('Realign: Estimate & Reslice: Mean Image', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','rmean'));
% matlabbatch{5}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70
%                                                           78 76 85];
% matlabbatch{5}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
% matlabbatch{5}.spm.spatial.normalise.write.woptions.interp = 4;
% matlabbatch{5}.spm.spatial.normalise.write.woptions.prefix = 'w';
% 
% 
% matlabbatch{9}.spm.stats.fmri_spec.dir = {OUT_DIR};
% matlabbatch{9}.spm.stats.fmri_spec.timing.units = 'secs';
% matlabbatch{9}.spm.stats.fmri_spec.timing.RT = 2;
% matlabbatch{9}.spm.stats.fmri_spec.timing.fmri_t = 16;
% matlabbatch{9}.spm.stats.fmri_spec.timing.fmri_t0 = 8;
% matlabbatch{9}.spm.stats.fmri_spec.sess(1).scans(1) = cfg_dep('Smooth: Smoothed Images', substruct('.','val', '{}',{6}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
% matlabbatch{9}.spm.stats.fmri_spec.sess(1).cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {}, 'orth', {});
% matlabbatch{9}.spm.stats.fmri_spec.sess(1).multi = {ONSETS_RUN_1};
% matlabbatch{9}.spm.stats.fmri_spec.sess(1).regress = struct('name', {}, 'val', {});
% matlabbatch{9}.spm.stats.fmri_spec.sess(1).multi_reg(1) = cfg_dep('Realign: Estimate & Reslice: Realignment Param File (Sess 1)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{1}, '.','rpfile'));
% matlabbatch{9}.spm.stats.fmri_spec.sess(1).hpf = 128;
% matlabbatch{9}.spm.stats.fmri_spec.sess(2).scans(1) = cfg_dep('Smooth: Smoothed Images', substruct('.','val', '{}',{7}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
% matlabbatch{9}.spm.stats.fmri_spec.sess(2).cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {}, 'orth', {});
% matlabbatch{9}.spm.stats.fmri_spec.sess(2).multi = {ONSETS_RUN_2};
% matlabbatch{9}.spm.stats.fmri_spec.sess(2).regress = struct('name', {}, 'val', {});
% matlabbatch{9}.spm.stats.fmri_spec.sess(2).multi_reg(1) = cfg_dep('Realign: Estimate & Reslice: Realignment Param File (Sess 2)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{2}, '.','rpfile'));
% matlabbatch{9}.spm.stats.fmri_spec.sess(2).hpf = 128;
% matlabbatch{9}.spm.stats.fmri_spec.sess(3).scans(1) = cfg_dep('Smooth: Smoothed Images', substruct('.','val', '{}',{8}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
% matlabbatch{9}.spm.stats.fmri_spec.sess(3).cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {}, 'orth', {});
% matlabbatch{9}.spm.stats.fmri_spec.sess(3).multi = {ONSETS_RUN_3};
% matlabbatch{9}.spm.stats.fmri_spec.sess(3).regress = struct('name', {}, 'val', {});
% matlabbatch{9}.spm.stats.fmri_spec.sess(3).multi_reg(1) = cfg_dep('Realign: Estimate & Reslice: Realignment Param File (Sess 3)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{3}, '.','rpfile'));
% matlabbatch{9}.spm.stats.fmri_spec.sess(3).hpf = 128;
% matlabbatch{9}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
% matlabbatch{9}.spm.stats.fmri_spec.bases.hrf.derivs = [1 0];
% matlabbatch{9}.spm.stats.fmri_spec.volt = 1;
% matlabbatch{9}.spm.stats.fmri_spec.global = 'None';
% matlabbatch{9}.spm.stats.fmri_spec.mthresh = 0.8;
% matlabbatch{9}.spm.stats.fmri_spec.mask = {''};
% matlabbatch{9}.spm.stats.fmri_spec.cvi = 'AR(1)';
% matlabbatch{10}.spm.stats.fmri_est.spmmat(1) = cfg_dep('fMRI model specification: SPM.mat File', substruct('.','val', '{}',{9}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
% matlabbatch{10}.spm.stats.fmri_est.write_residuals = 0;
% matlabbatch{10}.spm.stats.fmri_est.method.Classical = 1;
% matlabbatch{11}.spm.stats.con.spmmat(1) = cfg_dep('Model estimation: SPM.mat File', substruct('.','val', '{}',{10}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
% matlabbatch{11}.spm.stats.con.consess{1}.tcon.name = 'pumps demean vs ctrl demean';
% matlabbatch{11}.spm.stats.con.consess{1}.tcon.weights = [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0];
% matlabbatch{11}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
% matlabbatch{11}.spm.stats.con.delete = 0;
% matlabbatch{12}.spm.stats.results.spmmat(1) = cfg_dep('Contrast Manager: SPM.mat File', substruct('.','val', '{}',{11}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
% matlabbatch{12}.spm.stats.results.conspec.titlestr = 'pumps demean vs ctrl demean';
% matlabbatch{12}.spm.stats.results.conspec.contrasts = Inf;
% matlabbatch{12}.spm.stats.results.conspec.threshdesc = 'none';
% matlabbatch{12}.spm.stats.results.conspec.thresh = 0.01;
% matlabbatch{12}.spm.stats.results.conspec.extent = 0;
% matlabbatch{12}.spm.stats.results.conspec.conjunction = 1;
% matlabbatch{12}.spm.stats.results.conspec.mask.none = 1;
% matlabbatch{12}.spm.stats.results.units = 1;
% matlabbatch{12}.spm.stats.results.export{1}.pdf = true;
% matlabbatch{12}.spm.stats.results.export{2}.tspm.basename = 'thresh_';
% matlabbatch{12}.spm.stats.results.export{3}.nidm.modality = 'FMRI';
% matlabbatch{12}.spm.stats.results.export{3}.nidm.refspace = 'ixi';
% matlabbatch{12}.spm.stats.results.export{3}.nidm.group.nsubj = 1;
% matlabbatch{12}.spm.stats.results.export{3}.nidm.group.label = 'subject';