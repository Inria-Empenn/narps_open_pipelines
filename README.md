# THE OPEN PIPELINE PROJECT

<p align="center">
	<img src="https://github.com/elodiegermani/narps_open_pipelines/blob/main/static/images/project_illustration.png"/> 
</p>

## Table of contents
   * [Project presentation](#project-presentation)
   * [Getting Started](#getting-started)
   	  * [Contents overview](#contents-overview)
	   	  * [src](#src)
	   	  * [data](#data)
   	  * [Install Docker container](#install-docker-container)
   	  * [Download data](#download-data)
   	  	  * [Original dataset](#original-dataset)
   	  	  * [Derived data](#derived-data)
	  	  * [Contributing](#contributing)
   * [References](#references)


## Project presentation

Neuroimaging workflows are highly flexible, leaving researchers with multiple possible options to analyze a dataset [(Carp, 2012)](https://www.frontiersin.org/articles/10.3389/fnins.2012.00149/full).
However, different analytical choices can cause variation in the results [(Botvinik-Nezer et al., 2020)](https://www.nature.com/articles/s41586-020-2314-9), leading to what was called a "vibration of effects" [(Ioannidis, 2008)](https://pubmed.ncbi.nlm.nih.gov/18633328/) or analytical variability. 

The open pipeline project aims at providing a overview of the possible pipelines that are used in the community to analyze a dataset. 

## Getting Started

[Fork and clone](https://docs.github.com/en/get-started/quickstart/fork-a-repo) this repository to your local machine. 

### Contents overview

#### `src`

This directory contains scripts of the reproduce pipelines along with notebooks and/or scripts to launch them. 

#### `data`

This directory is made to contain data that will be used by scripts/notebooks stored in the `src` directory and to contain the results (intermediate results and final data) of those scripts. 

Instructions to download data are available [below](#download-data).

### Install Docker container

*Coming soon*

### Data Download Instructions
The dataset used for the `/src/reproduction_*.ipynb` can be downloaded using one of the three options below. For the scripts to work as intended, the dataset **MUST** be stored in `data/original`.

On your local copy, place yourself in `data/original` directory before running the commands below.
#### Original Dataset
**Option 1: with DataLad (Recommended)**
1. Install DataLad and it's dependencies from **[here](http://handbook.datalad.org/en/latest/intro/installation.html)**, if you don't have it installed already.
2. After installation, run one of the following commands:

    `datalad install https://github.com/OpenNeuroDatasets/ds001734.git`

    **OR**

    `datalad install ///openneuro/ds001734`

The `datalad install` command only downloads the metadata associated with the dataset. To download the actual files run 
`datalad get ./*` and if you only want parts of the data, replace the `*` by the paths to the desired files.

**Option 2: with Node.js**

Using [@openneuro/cli](https://www.npmjs.com/package/@openneuro/cli) you can download this dataset from the command line using [Node.js](https://nodejs.org/en/download/). This method is good for larger datasets or unstable connections, but has known issues on Windows.

`openneuro download --snapshot 1.0.5 ds001734 ds001734/`

This will download to `ds001734/` in the current directory. If your download is interrupted and you need to retry, rerun the command to resume the download.

**Option 3: from S3**

The most recently published snapshot can be downloaded from S3. This method is best for larger datasets or unstable connections. This example uses [AWS CLI](https://aws.amazon.com/cli/).

`aws s3 sync --no-sign-request s3://openneuro.org/ds001734 ds001734-download/`

File containing pipeline description is available in `/data/original`.


Download Instructions Source: https://openneuro.org/datasets/ds001734/versions/1.0.5/download
#### Derived data

Derived data such as original stat maps from teams and reproduced stat maps can be downloaded from [NeuroVault](www.neurovault.org) [(Gorgolewski & al, 2015)](https://www.frontiersin.org/articles/10.3389/fninf.2015.00008/full). 

*Coming soon*

### Contributing 

Follow the guidelines in [CONTRIBUTING.md](https://github.com/elodiegermani/open_pipeline/blob/main/CONTRIBUTING.md)

## References

1. [Botvinik-Nezer, R. et al. (2020), ‘Variability in the analysis of a single neuroimaging dataset by many teams’, Nature.](https://www.nature.com/articles/s41586-020-2314-9)
2. [Carp, J. et al. (2012), ‘On the Plurality of (Methodological) Worlds: Estimating the Analytic Flexibility of fMRI Experiments’, Frontiers in Neuroscience.](https://www.frontiersin.org/articles/10.3389/fnins.2012.00149/full)
3. [Gorgolewski, K.J. et al. (2015), ‘NeuroVault.org: a web-based repository for collecting and sharing unthresholded statistical maps of the human brain’ Frontiers in Neuroinformatics.](https://www.frontiersin.org/articles/10.3389/fninf.2015.00008/full)
4. [Ioannidis, J.P.A. (2008), ‘Why Most Discovered True Associations Are Inflated’, Epidemiology.](https://pubmed.ncbi.nlm.nih.gov/18633328/)
