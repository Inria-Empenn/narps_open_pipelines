# More information about the NARPS study

This page aims at summarizing the NARPS study [(Botvinik-Nezer et al., 2020)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7771346/) for the future contributors of NARPS Open Pipelines.

In the following, the citations come from the associated data article : [Botvinik-Nezer, R. et al. (2019), 'fMRI data of mixed gambles from the Neuroimaging Analysis Replication and Prediction Study', Scientific Data](https://www.nature.com/articles/s41597-019-0113-7).

## Context

> In the Neuroimaging Analysis Replication and Prediction Study (NARPS), we aim to provide the first scientific evidence on the variability of results across analysis teams in neuroscience.

From this starting point, 70 teams were asked to analyse the same dataset, providing their methods and results to be later analysed and compared. Nine hypotheses where to be tested and a binary decision for each one had to be reported, whether it was significantly supported based on a whole-brain analysis.

## The data

NARPS is based on a dataset of task-fMRI with 108 participants, 4 runs for each.

> For each participant, the dataset includes an anatomical (T1 weighted) scan and fMRI as well as behavioral data from four runs of the task. The dataset is shared through OpenNeuro and is formatted according to the Brain Imaging Data Structure (BIDS) standard. Data pre-processed with fMRIprep and quality control reports are also publicly shared.

### Raw functional volumes

> Each run consists of 64 trials performed during an fMRI scanning run lasting 453 seconds and comprising 453 volumes (given the repetition time of one second).

For each participant, the associated data (4D volumes) is :
* `data/original/ds001734/sub-*/func/sub-*_task-MGT_run-01_bold.nii.gz`
* `data/original/ds001734/sub-*/func/sub-*_task-MGT_run-02_bold.nii.gz`
* `data/original/ds001734/sub-*/func/sub-*_task-MGT_run-03_bold.nii.gz`
* `data/original/ds001734/sub-*/func/sub-*_task-MGT_run-04_bold.nii.gz`

### Event data

> On each trial, participants were presented with a mixed gamble entailing an equal 50% chance of gaining one amount of money or losing another amount.

For each participant, the associated data (events files) is :
* `data/original/ds001734/sub-*/func/sub-*_task-MGT_run-01_events.tsv`
* `data/original/ds001734/sub-*/func/sub-*_task-MGT_run-02_events.tsv`
* `data/original/ds001734/sub-*/func/sub-*_task-MGT_run-03_events.tsv`
* `data/original/ds001734/sub-*/func/sub-*_task-MGT_run-04_events.tsv`

This file contains the onsets, response time, and response of the participant, as well as the amount of money proposed (gain and loss) for each trial.

### Preprocessed data

> The pre-processed data included in this dataset were preprocessed using fMRIprep version 1.1.4, which is based on Nipype 1.1.1

For each participant, the associated data (preprocessed volumes, confounds, brain masks, ...) is under :
* `data/original/ds001734/derivatives/fmriprep/sub-*/func/`

Some teams chose to use this pre-processed data directly as inputs for the statistical analysis, while other performed their own method of pre-processing.

### Task-related data

The associated data (task and dataset description) is :
* `data/original/ds001734/T1w.json`
* `data/original/ds001734/task-MGT_bold.json`
* `data/original/ds001734/task-MGT_sbref.json`
* `data/original/ds001734/dataset_description.json`

> [!TIP]
> The `narps_open.data.task` module helps parsing this data.

Furthermore, the participants were assigned to a condition, either *equalRange* or *equalIndifference* :

> Possible gains ranged between 10 and 40 ILS (in increments of 2 ILS) in the equal indifference condition or 5–20 ILS (in increments of 1 ILS) in the equal range condition, while possible losses ranged from 5–20 ILS (in increments of 1 ILS) in both conditions.

The repartition is stored in :
* `data/original/ds001734/participants.tsv`

> [!TIP]
> The `narps_open.data.participants` module helps parsing this data.

## The outputs

Each of the team participating in NARPS had to provide a COBIDS-compliant *textual description* of its analysis of the dataset, as well as the results from it.

All the descriptions are gathered in the [analysis_pipelines_for_analysis.xlsx](https://github.com/poldrack/narps/blob/1.0.1/ImageAnalyses/metadata_files/analysis_pipelines_for_analysis.xlsx) in the NARPS repository on GitHub.

> [!TIP]
> We developed a tool to easily parse this file, see [docs/description.md](docs/description.md) for more details on how to use it.

Results data submitted by all the teams is available [on Zenodo](https://zenodo.org/records/3528329#.Y7_H1bTMKBT)

> [!TIP]
> This data is included in the NARPS Open Pipelines repository under [data/results](data/results), and we developed a tool to access it easily : see the dedicated section in [docs/data.md](docs/data.md#results-from-narps-teams) for more details.

## Useful resources

* The website dedicated to the study - [www.narps.info](https://www.narps.info/)
* The article - [Botvinik-Nezer, R. et al. (2020), 'Variability in the analysis of a single neuroimaging dataset by many teams', Nature](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7771346/)
* The associated data article - [Botvinik-Nezer, R. et al. (2019), 'fMRI data of mixed gambles from the Neuroimaging Analysis Replication and Prediction Study', Scientific Data](https://www.nature.com/articles/s41597-019-0113-7)
* The GitHub page for the NARPS repository. This code was used to generate the published results - [github.com/poldrack/narps](https://github.com/poldrack/narps)
* The snapshot of this code base [on Zenodo](https://zenodo.org/records/3709273#.Y2jVkCPMIz4)
* The dataset [on OpenNeuro](https://openneuro.org/datasets/ds001734/versions/1.0.5)
* Results data submitted by all the teams [on Zenodo](https://zenodo.org/records/3528329#.Y7_H1bTMKBT)
