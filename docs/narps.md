# More information about the NARPS study

This page aims at summarizing the NARPS study (Botvinik-Nezer et al., 2020) for the future contributors of NARPS Open Pipelines.

## Context

The global context / problem : analytical variability



## The main idea behind the study

Asking 70 teams to analyse the same dataset

## The data

NARPS is based on a dataset of task-fMRI with 108 participants, 4 runs for each.

### Raw functional volumes

> Each run consists of 64 trials performed during an fMRI scanning run lasting 453 seconds and comprising 453 volumes (given the repetition time of one second).

For each participant, the associated data (4D volumes) is :
`data/original/ds001734/sub-*/func/sub-*_task-MGT_run-01_bold.nii.gz`
`data/original/ds001734/sub-*/func/sub-*_task-MGT_run-02_bold.nii.gz`
`data/original/ds001734/sub-*/func/sub-*_task-MGT_run-03_bold.nii.gz`
`data/original/ds001734/sub-*/func/sub-*_task-MGT_run-04_bold.nii.gz`

### Event data

> On each trial, participants were presented with a mixed gamble entailing an equal 50% chance of gaining one amount of money or losing another amount.

For each participant, the associated data (events files) is :
`data/original/ds001734/sub-*/func/sub-*_task-MGT_run-01_events.tsv`
`data/original/ds001734/sub-*/func/sub-*_task-MGT_run-02_events.tsv`
`data/original/ds001734/sub-*/func/sub-*_task-MGT_run-03_events.tsv`
`data/original/ds001734/sub-*/func/sub-*_task-MGT_run-04_events.tsv`

This file contains the onsets, response time, and response of the participant, as well as the amount of money proposed (gain and loss) for each trial.

> The pre-processed data included in this dataset were preprocessed using fMRIprep version 1.1.4, which is based on Nipype 1.1.1

For each participant, the associated data (preprocessed volumes, confounds, brain masks, ...) is under :
`data/original/ds001734/derivatives/fmriprep/sub-*/func/`

### Task-related data

The associated data (task and dataset description) is :
`data/original/ds001734/T1w.json`
`data/original/ds001734/task-MGT_bold.json`
`data/original/ds001734/task-MGT_sbref.json`
`data/original/ds001734/dataset_description.json`

Furthermore, the participants were assigned to a condition, either *equalRange* or *equalIndifference* :

> Possible gains ranged between 10 and 40 ILS (in increments of 2 ILS) in the equal indifference condition or 5–20 ILS (in increments of 1 ILS) in the equal range condition, while possible losses ranged from 5–20 ILS (in increments of 1 ILS) in both conditions.

The repartition is stored in :
`data/original/ds001734/participants.tsv`

## The outputs

textual descriptions + team results
](https://github.com/poldrack/narps/blob/1.0.1/ImageAnalyses/metadata_files/analysis_pipelines_for_analysis.xlsx)

Data submitted by all participants in the Neuroimaging Analysis Replication and Prediction Study, along with results from prediction markets and metadata for analysis pipelines.
https://zenodo.org/records/3528329#.Y7_H1bTMKBT

## Useful resources

* The website dedicated to the study - [www.narps.info](https://www.narps.info/)
* The article - [Botvinik-Nezer, R. et al. (2020), 'Variability in the analysis of a single neuroimaging dataset by many teams', Nature](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7771346/)
* The associated data article - [Botvinik-Nezer, R. et al. (2019), 'fMRI data of mixed gambles from the Neuroimaging Analysis Replication and Prediction Study', Scientific Data](https://www.nature.com/articles/s41597-019-0113-7)
* The GitHub page for the NARPS repository. This code was used to generate the published results - [github.com/poldrack/narps](https://github.com/poldrack/narps)
* The snapshot of this code base [on Zenodo](https://zenodo.org/records/3709273#.Y2jVkCPMIz4)
* https://openneuro.org/datasets/ds001734/versions/1.0.5

Raw and preprocessed fMRI data of two versions of the mixed gambles task, from the Neuroimaging Analysis Replication and Prediction Study
https://openneuro.org/datasets/ds001734/versions/1.0.5