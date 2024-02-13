% first-level job ----

% $ narps_description -t R5K7 -d preprocessing --json

clear matlabbatch;

% 1) motion correction

% "motion_correction": "SPM12, Realign & Unwarp using the phase map 
% generated from the fieldmap data with SPM12 Fieldmap Toolbox v2.1 
% (default options). \nOther than defaults: \n- estimation: quality 0.95, 
% separation 3, two-pass procedure (i. registering to 1st scan, ii. 
% registering to mean image), interpolation 7th degree B-spline; \n- unwarp
% & reslice: interpolation 7th degree B-spline ",

% *** Create VDM map from field maps
matlabbatch{1}.spm.tools.fieldmap.calculatevdm.subj.data.presubphasemag.phase = {
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/fmap/sub-001_phasediff.nii,1'};
matlabbatch{1}.spm.tools.fieldmap.calculatevdm.subj.data.presubphasemag.magnitude = {
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/fmap/sub-001_magnitude1.nii,1'};
matlabbatch{1}.spm.tools.fieldmap.calculatevdm.subj.defaults.defaultsval.et = [0.00492 0.00738];
matlabbatch{1}.spm.tools.fieldmap.calculatevdm.subj.defaults.defaultsval.maskbrain = 0;
matlabbatch{1}.spm.tools.fieldmap.calculatevdm.subj.defaults.defaultsval.blipdir = -1;
matlabbatch{1}.spm.tools.fieldmap.calculatevdm.subj.defaults.defaultsval.tert = 29.15;
matlabbatch{1}.spm.tools.fieldmap.calculatevdm.subj.defaults.defaultsval.epifm = 0;
matlabbatch{1}.spm.tools.fieldmap.calculatevdm.subj.session(1).epi = {
    '/Users/camaumet/Softs/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-01_bold.nii,1'};
matlabbatch{1}.spm.tools.fieldmap.calculatevdm.subj.session(2).epi = {
    '/Users/camaumet/Softs/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-02_bold.nii,1'};
matlabbatch{1}.spm.tools.fieldmap.calculatevdm.subj.session(3).epi = {
    '/Users/camaumet/Softs/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-03_bold.nii,1'};
matlabbatch{1}.spm.tools.fieldmap.calculatevdm.subj.session(4).epi = {
    '/Users/camaumet/Softs/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-04_bold.nii,1'};
matlabbatch{1}.spm.tools.fieldmap.calculatevdm.subj.matchvdm = 1;
matlabbatch{1}.spm.tools.fieldmap.calculatevdm.subj.writeunwarped = 0;
matlabbatch{1}.spm.tools.fieldmap.calculatevdm.subj.matchanat = 0;

% --> This generates 4 files whose name start with vdm5_
% In the following we will denote them by vdm5_runXX.extension


matlabbatch{end+1}.spm.spatial.realignunwarp.data(1).scans = {
    % Note: as many volumes as 3d volume in the fmri 4D image    
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-01_bold.nii,1'
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-01_bold.nii,2'
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-01_bold.nii,3'};
matlabbatch{end}.spm.spatial.realignunwarp.data(1).pmscan = {'ABS_PATH/vdm5_run01.nii'};
matlabbatch{end}.spm.spatial.realignunwarp.data(2).scans = {
    '/Users/camaumet/Softs/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-02_bold.nii,1'
    '/Users/camaumet/Softs/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-02_bold.nii,2'
    '/Users/camaumet/Softs/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-02_bold.nii,3'};
matlabbatch{end}.spm.spatial.realignunwarp.data(2).pmscan = {'ABS_PATH/vdm5_run02.nii'};
matlabbatch{end}.spm.spatial.realignunwarp.data(3).scans = {
    '/Users/camaumet/Softs/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-03_bold.nii,1'
    '/Users/camaumet/Softs/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-03_bold.nii,2'
    '/Users/camaumet/Softs/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-02_bold.nii,3'};
matlabbatch{end}.spm.spatial.realignunwarp.data(3).pmscan = {'ABS_PATH/vdm5_run03.nii'};
matlabbatch{end}.spm.spatial.realignunwarp.data(4).scans = {
    '/Users/camaumet/Softs/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-04_bold.nii,1'
    '/Users/camaumet/Softs/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-04_bold.nii,2'
    '/Users/camaumet/Softs/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-04_bold.nii,3'};
matlabbatch{end}.spm.spatial.realignunwarp.data(4).pmscan = {'ABS_PATH/vdm5_run04.nii'};
matlabbatch{end}.spm.spatial.realignunwarp.eoptions.quality = 0.95;
matlabbatch{end}.spm.spatial.realignunwarp.eoptions.sep = 3;
matlabbatch{end}.spm.spatial.realignunwarp.eoptions.rtm = 1;
matlabbatch{end}.spm.spatial.realignunwarp.eoptions.einterp = 7;
matlabbatch{end}.spm.spatial.realignunwarp.uwroptions.rinterp = 7;

% --> Here we get 4 files usub-001_task-MGT_run-XX_bold.nii as well as 1
% unwarped_mean_image.nii


% 2) intersubject registration (normalization)

% Note: we need to do segmentation first
matlabbatch{end+1}.spm.tools.oldseg.data = {
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/anat/sub-001_T1w.nii,1'};

% --> Here we get a c1sub-001_T1w.nii file that is the grey matter
% probability map and transfo.nii file (not sure it is a .nii) that is the
% calculated transformation between anat and standardized space

% Coreg each sbref onto mean unwarp
matlabbatch{end+1}.spm.spatial.coreg.estimate.ref(1) = {
    'ABS_PATH/unwarped_mean_image.nii'
    };
matlabbatch{end}.spm.spatial.coreg.estimate.source(1) = {
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-01_sbref.nii'};
matlabbatch{end}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';

matlabbatch{end+1}.spm.spatial.coreg.estimate.ref(1) = {
    'ABS_PATH/unwarped_mean_image.nii'
    };
matlabbatch{end}.spm.spatial.coreg.estimate.source(1) = {
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-02_sbref.nii'};
matlabbatch{end}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';

matlabbatch{end+1}.spm.spatial.coreg.estimate.ref(1) = {
    'ABS_PATH/unwarped_mean_image.nii'
    };
matlabbatch{end}.spm.spatial.coreg.estimate.source(1) = {
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-03_sbref.nii'};
matlabbatch{end}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';

matlabbatch{end+1}.spm.spatial.coreg.estimate.ref(1) = {
    'ABS_PATH/unwarped_mean_image.nii'
    };
matlabbatch{end}.spm.spatial.coreg.estimate.source(1) = {
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-04_sbref.nii'};
matlabbatch{end}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';

% --> HERE WE get 4 file ssub-001_task-MGT_run-XX_sbref.nii (that
% keeps the same name as before *but* the header has been modified to apply
% the coregistration'

matlabbatch{end+1}.spm.spatial.coreg.estimate.ref(1) = {
    'ABS_PATH/c1sub-001_T1w.nii'
};
matlabbatch{end}.spm.spatial.coreg.estimate.source(1) = {
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-01_sbref.nii'
    };
matlabbatch{end}.spm.spatial.coreg.estimate.other = {'usub-001_task-MGT_run-01_bold.nii'
    'usub-001_task-MGT_run-02_bold.nii'
    'usub-001_task-MGT_run-03_bold.nii'
    'usub-001_task-MGT_run-04_bold.nii'
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-02_sbref.nii'
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-03_sbref.nii'
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-04_sbref.nii'
};
matlabbatch{end}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';


% We apply the transformation to standardized space

matlabbatch{end+1}.spm.spatial.normalise.write.subj.def = {'transfo.nii'};
matlabbatch{end}.spm.spatial.normalise.write.subj.resample = {
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-01_bold.nii'
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-02_bold.nii'
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-03_bold.nii'
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-04_bold.nii'
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-02_sbref.nii'
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-03_sbref.nii'
    'ABS_PATH/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-04_sbref.nii'
};

% 3) spatial smoothing
