# Access the descriptions of NARPS teams pipelines

The file `narps_open/utils/description/analysis_pipelines_full_descriptions.tsv` contains the description provided by each team participating to NARPS.
It is a convertion into tsv format (tab-separated values) of the [original .xlsx file published in NARPS](https://github.com/poldrack/narps/blob/1.0.1/ImageAnalyses/metadata_files/analysis_pipelines_for_analysis.xlsx
), which allows easier parsing with python.

The file `narps_open/utils/description/analysis_pipelines_derived_descriptions.tsv` contains for each team a set of programatically usable data based on the textual descriptions of the previous file. This data is available in the `derived` sub dictionary (see examples hereafter).

The class `TeamDescription` of module `narps_open.utils.description` acts as a parser for these two files.

You can also use the command-line tool as so. Option `-t` is for the team id, option `-d` allows to print only one of the sub parts of the description among : `general`, `exclusions`, `preprocessing`, `analysis`, and `categorized_for_analysis`.

```bash
python narps_open/utils/description -h
# usage: __init__.py [-h] -t TEAM [-d {general,exclusions,preprocessing,analysis,categorized_for_analysis,derived}]
#
# Get description of a NARPS pipeline.
#
# options:
#   -h, --help            show this help message and exit
#   -t TEAM, --team TEAM  the team ID
#   -d {general,exclusions,preprocessing,analysis,categorized_for_analysis,derived}, --dictionary {general,exclusions,preprocessing,analysis,categorized_for_analysis,derived}
#                         the sub dictionary of team description

python narps_open/utils/description -t 2T6S -d general
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
```

Of course the `narps_open.utils.description` module is accessible programmatically, here is an example on how to use it:

```python
from narps_open.utils.description import TeamDescription
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
