# Access the descriptions of NARPS teams pipelines

The file `narps_open/data/description/analysis_pipelines_full_descriptions.tsv` contains the description provided by each team participating to NARPS.
It is a conversion into tsv format (tab-separated values) of the [original .xlsx file published in NARPS](https://github.com/poldrack/narps/blob/1.0.1/ImageAnalyses/metadata_files/analysis_pipelines_for_analysis.xlsx
), which allows easier parsing with python.

The file `narps_open/data/description/analysis_pipelines_derived_descriptions.tsv` contains for each team a set of programmatically usable data based on the textual descriptions of the previous file. This data is available in the `derived` sub dictionary (see examples hereafter).

The file `narps_open/data/description/analysis_pipelines_comments.tsv` contains for each team a set of comments made by the NARPS Open Pipelines team about reproducibility and exclusions of the pipeline. This data is available in the `comments` sub dictionary (see examples hereafter).

The class `TeamDescription` of module `narps_open.data.description` acts as a parser for these two files.

You can use the command-line tool as so. Option `-t` is for the team id, option `-d` allows to print only one of the sub parts of the description among : `general`, `exclusions`, `preprocessing`, `analysis`, `categorized_for_analysis`, `derived`, and `comments`. Options `--json` and `--md` allow to choose the export format you prefer between JSON and Markdown.

```bash
python narps_open/data/description -h
# usage: __init__.py [-h] -t TEAM [-d {general,exclusions,preprocessing,analysis,categorized_for_analysis,derived,comments}]
#
# Get description of a NARPS pipeline.
#
# options:
#   -h, --help            show this help message and exit
#   -t TEAM, --team TEAM  the team ID
#   -d {general,exclusions,preprocessing,analysis,categorized_for_analysis,derived,comments}, --dictionary {general,exclusions,preprocessing,analysis,categorized_for_analysis,derived,comments}
#                         the sub dictionary of team description
#  --json                output team description as JSON
#  --md                  output team description as Markdown

python narps_open/data/description -t 2T6S --json
# {
#     "general.teamID": "2T6S",
#     "general.NV_collection_link": "https://neurovault.org/collections/4881/",
#     "general.results_comments": "NA",
#     "general.preregistered": "No",
#     "general.link_preregistration_form": "We did not pre-register our analysis.",
#     "general.regions_definition": "We employed the pre-hypothesized brain regions (vmPFC, vSTR, and amygdala) from Barta, McGuire, and Kable (2010, Neuroimage). Specific MNI coordinates are:\nvmPFC: x = 2, y = 46, z = -8\nleft vSTR: x = -12, y = 12, z = -6, right vSTR = x = 12, y = 10, z = -6\n(right) Amygdala: x = 24, y = -4, z = -18",
#     "general.softwares": "SPM12 , \nfmriprep 1.1.4",
#     "exclusions.n_participants": "108",
#     "exclusions.exclusions_details": "We did not exclude any participant in the analysis",
#     "preprocessing.used_fmriprep_data": "Yes",
#     "preprocessing.preprocessing_order": "We used the provided preprocessed data by fMRIPprep 1.1.4 (Esteban, Markiewicz, et al. (2018); Esteban, Blair, et al. (2018); RRID:SCR_016216), which is based on Nipype 1.1.1 (Gorgolewski et al. (2011); Gorgolewski et al. (2018); RRID:SCR_002502) and we additionally conducted a spatial smoothing using the provided preprocessed data set and SPM12. Here, we attach the preprocessing steps described in the provided data set. \nAnatomical data preprocessing\nThe T1-weighted (T1w) image was corrected for intensity non-uniformity (INU) using N4BiasFieldCorrection (Tustison et al. 2010, ANTs 2.2.0), and used as T1w-reference throughout the workflow. The T1w-reference was then skull-stripped using antsBrainExtraction.sh (ANTs 2.2.0), using OASIS as target template. Brain surfaces we
# ...

python narps_open/data/description -t 2T6S -d general --json
# {
#    "teamID": "2T6S",
#    "NV_collection_link": "https://neurovault.org/collections/4881/",
#    "results_comments": "NA",
#    "preregistered": "No",
#    "link_preregistration_form": "We did not pre-register our analysis.",
#    "regions_definition": "We employed the pre-hypothesized brain regions (vmPFC, vSTR, and amygdala) from Barta, McGuire, and Kable (2010, Neuroimage). Specific MNI coordinates are:\nvmPFC: x = 2, y = 46, z = -8\nleft vSTR: x = -12, y = 12, z = -6, right vSTR = x = 12, y = 10, z = -6\n(right) Amygdala: x = 24, y = -4, z = -18",
#    "softwares": "SPM12 , \nfmriprep 1.1.4",
#    "general_comments": "NA"
# }

python narps_open/data/description -t 2T6S --md
# # NARPS team description : 2T6S
# ## General
# * `teamID` : 2T6S
# * `NV_collection_link` : https://neurovault.org/collections/4881/
# * `results_comments` : NA
# * `preregistered` : No
# * `link_preregistration_form` : We did not pre-register our analysis.
# * `regions_definition` : We employed the pre-hypothesized brain regions (vmPFC, vSTR, and amygdala) from Barta, McGuire, and Kable (2010, Neuroimage). Specific MNI coordinates are:
# vmPFC: x = 2, y = 46, z = -8
# left vSTR: x = -12, y = 12, z = -6, right vSTR = x = 12, y = 10, z = -6
# (right) Amygdala: x = 24, y = -4, z = -18
# * `softwares` : SPM12 , 
# fmriprep 1.1.4
# * `general_comments` : NA
# ## Exclusions
# * `n_participants` : 108
# * `exclusions_details` : We did not exclude any participant in the analysis
# ## Preprocessing
# * `used_fmriprep_data` : Yes
# * `preprocessing_order` : We used the provided preprocessed data by fMRIPprep 1.1.4 (Esteban, Markiewicz, et al. (2018); Esteban, Blair, et al. (2018); RRID:SCR_016216), which is based on Nipype 1.1.1 (Gorgolewski et al. (2011); Gorgolewski et al. (2018); RRID:SCR_002502) and we additionally conducted a spatial smoothing using the provided preprocessed data set and SPM12. Here, we attach the preprocessing steps described in the provided data set. 
# Anatomical data preprocessing
# ...
```

Of course the `narps_open.data.description` module is accessible programmatically, here is an example on how to use it:

```python
from narps_open.data.description import TeamDescription
description = TeamDescription('2T6S') # Set the id of the team here
# Access the object as a dict
print(description)
description['general.teamID']
# Access sub dictionaries
description.general
description.exclusions
description.preprocessing
description.analysis
description.categorized_for_analysis
description.derived
description.comments
# Access values of sub dictionaries
description.general['teamID']
# Other keys in general are: ['teamID', 'NV_collection_link', 'results_comments', 'preregistered', 'link_preregistration_form', 'regions_definition', 'softwares', 'general_comments']
description.exclusions['n_participants']
# Other keys in exclusions are: ['n_participants', 'exclusions_details']
description.preprocessing['motion']
# Other keys in preprocessing are: ['used_fmriprep_data', 'preprocessing_order', 'brain_extraction', 'segmentation', 'slice_time_correction', 'motion_correction', 'motion', 'gradient_distortion_correction', 'intra_subject_coreg', 'distortion_correction', 'inter_subject_reg', 'intensity_correction', 'intensity_normalization', 'noise_removal', 'volume_censoring', 'spatial_smoothing', 'preprocessing_comments']
description.analysis['RT_modeling']
# Other keys in analysis are: ['data_submitted_to_model', 'spatial_region_modeled', 'independent_vars_first_level', 'RT_modeling', 'movement_modeling', 'independent_vars_higher_level', 'model_type', 'model_settings', 'inference_contrast_effect', 'search_region', 'statistic_type', 'pval_computation', 'multiple_testing_correction', 'comments_analysis']
description.categorized_for_analysis['analysis_SW_with_version']
# Other keys in categorized_for_analysis are: ['region_definition_vmpfc', 'region_definition_striatum', 'region_definition_amygdala', 'analysis_SW', 'analysis_SW_with_version', 'smoothing_coef', 'testing', 'testing_thresh', 'correction_method', 'correction_thresh_']
description.derived['n_participants']
# Other keys in derived are: ['n_participants', 'excluded_participants', 'func_fwhm', 'con_fwhm']
```
