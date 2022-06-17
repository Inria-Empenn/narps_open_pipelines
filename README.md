# THE OPEN PIPELINE PROJECT

<p align="center">
	<img src="https://github.com/elodiegermani/narps_open_pipelines/blob/main/static/images/project_illustration.png"/> 
</p>

## Table of contents
   * [Project presentation](#project-presentation)
   * [To start](#to-start)
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

Neuroimaging workflows are highly flexible, leaving researchers with multiple possible options to analyze a dataset (Carp, 2012).
However, different analytical choices can cause variation in the results (Botvinik-Nezer et al., 2020), leading to what was called a "vibration of effects" (Ioannidis, 2008) also known as analytical variability. 

**The goal of the NARPS open pipeline project is to create a codebase reproducing the 70 pipelines of the NARPS project (Botvinik-Nezer et al., 2020) and share this as an open resource for the community**. 

To perform the reproduction, we are lucky to be able to use the description provided by each team available here.

## To start 

### Contents overview

#### `src`

This directory contains scripts of the reproduce pipelines along with notebooks and/or scripts to launch them. 

#### `data`

This directory is made to contain data that will be used by scripts/notebooks stored in the `src` directory and to contain the results (intermediate results and final data) of those scripts. 

Instructions to download data are available [below](#download-data).

### Install Docker container

*Coming soon*

### Download data 

#### Original dataset

File containing pipeline description is available in `/data/original`.

The dataset used for the `/src/reproduction_*.ipynb` notebooks can be downloaded [**here**](https://openneuro.org/datasets/ds001734/versions/1.0.5).

The data must be stored in a directory inside the `data/original` directory. 

I recommand to download it with **Datalad**. If you want to use it, install [**Datalad**](http://handbook.datalad.org/en/latest/intro/installation.html#install), place yourself in the `data/original` directory and run `datalad install ///openneuro/ds001734`.
After, you can download all the files by running `datalad get ./*` and if you only want parts of the data, replace the * by the paths to the desired files. 

#### Derived data

Derived data such as original stat maps from teams and reproduced stat maps can be downloaded from [NeuroVault](www.neurovault.org) (Gorgolewski & al, 2015). 

*Coming soon*

### Contributing 

Follow the guidelines in [CONTRIBUTING.md](https://github.com/elodiegermani/open_pipeline/blob/main/CONTRIBUTING.md)

## References

- Botvinik-Nezer, R. et al. (2020), ‘Variability in the analysis of a single neuroimaging dataset by many teams’, Nature.
- Carp, J. et al. (2012), ‘On the Plurality of (Methodological) Worlds: Estimating the Analytic Flexibility of fMRI Experiments’, Frontiers in Neuroscience. ;
- Gorgolewski, K.J. et al. (2015), ‘NeuroVault.org: a web-based repository for collecting and sharing unthresholded statistical maps of the human brain’ Frontiers in Neuroinformatics. ;
- Ioannidis, J.P.A. (2008), ‘Why Most Discovered True Associations Are Inflated’, Epidemiology. ;
