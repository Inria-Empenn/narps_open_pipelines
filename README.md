# The NARPS Open Pipelines project

<p align="center">
	<img src="assets/images/project_illustration.png"/> 
</p>

## Table of contents

- [Project presentation](#project-presentation)
- [Getting Started](#getting-started)
	- [Contents overview](#contents-overview)
	- [Installation](#installation)
	- [Contributing](#contributing)
- [References](#references)

## Project presentation

Neuroimaging workflows are highly flexible, leaving researchers with multiple possible options to analyze a dataset [(Carp, 2012)](https://www.frontiersin.org/articles/10.3389/fnins.2012.00149/full).
However, different analytical choices can cause variation in the results [(Botvinik-Nezer et al., 2020)](https://www.nature.com/articles/s41586-020-2314-9), leading to what was called a "vibration of effects" [(Ioannidis, 2008)](https://pubmed.ncbi.nlm.nih.gov/18633328/) also known as analytical variability. 

**The goal of the NARPS open pipeline project is to create a codebase reproducing the 70 pipelines of the NARPS project (Botvinik-Nezer et al., 2020) and share this as an open resource for the community**. 

To perform the reproduction, we are lucky to be able to use the description provided by each team available [here](https://github.com/poldrack/narps/blob/1.0.1/ImageAnalyses/metadata_files/analysis_pipelines_for_analysis.xlsx). 
We also created a shared spreadsheet that can be use to add comments on pipelines: the ones that are already reproduced, the ones that are not reproducible with NiPype... You can find it [here](https://docs.google.com/spreadsheets/d/1FU_F6kdxOD4PRQDIHXGHS4zTi_jEVaUqY_Zwg0z6S64/edit?usp=sharing).

## Getting Started

Follow the instructions of [INSTALL.md](/INSTALL.md) to start with the NARPS open pipelines project.

### Contents overview

#### `narps_open` :snake: :package:

This directory contains the Python package with all the pipelines logic.

#### `data` :brain:

This directory is made to contain data that will be used by the pipelines, as well as the (intermediate or final) results data.

Instructions to download data are available in [INSTALL.md](/INSTALL.md#data-download-instructions).

#### `docs` :blue_book:

This directory the documentation for the project. Start browsing it with the entry point [docs/README.md](/docs/README.md)

#### `examples` :orange_book:

This directory contains notebooks examples to launch of the reproduced pipelines.

#### `tests` :microscope:

This directory contains the tests of the narps_open package.

### Installation

To get the pipelines running, please follow the installation steps in [INSTALL.md](/INSTALL.md)

### Contributing 

:wave: Any help is welcome ! Follow the guidelines in [CONTRIBUTING.md](/CONTRIBUTING.md) if you wish to get involed !

## References

1. [Botvinik-Nezer, R. et al. (2020), ‘Variability in the analysis of a single neuroimaging dataset by many teams’, Nature.](https://www.nature.com/articles/s41586-020-2314-9)
2. [Carp, J. et al. (2012), ‘On the Plurality of (Methodological) Worlds: Estimating the Analytic Flexibility of fMRI Experiments’, Frontiers in Neuroscience.](https://www.frontiersin.org/articles/10.3389/fnins.2012.00149/full)
3. [Gorgolewski, K.J. et al. (2015), ‘NeuroVault.org: a web-based repository for collecting and sharing unthresholded statistical maps of the human brain’ Frontiers in Neuroinformatics.](https://www.frontiersin.org/articles/10.3389/fninf.2015.00008/full)
4. [Ioannidis, J.P.A. (2008), ‘Why Most Discovered True Associations Are Inflated’, Epidemiology.](https://pubmed.ncbi.nlm.nih.gov/18633328/)

## Funding

This project is supported by Région Bretagne (Boost MIND). 
