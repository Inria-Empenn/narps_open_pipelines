# The NARPS Open Pipelines project

<p align="center">
	<img src="assets/images/project_illustration.png"/> 
</p>

<p align="center">
    <a href="https://github.com/Inria-Empenn/narps_open_pipelines/actions/workflows/unit_tests.yml" alt="Unit tests status">
        <img src="https://img.shields.io/github/actions/workflow/status/Inria-Empenn/narps_open_pipelines/unit_tests.yml?label=unit%20tests" /></a>
    <a href="https://github.com/Inria-Empenn/narps_open_pipelines/actions/workflows/code_quality.yml" alt="Code quality status">
        <img src="https://img.shields.io/github/actions/workflow/status/Inria-Empenn/narps_open_pipelines/code_quality.yml?label=code%20quality" /></a>
    <a href="https://github.com/Inria-Empenn/narps_open_pipelines/graphs/contributors" alt="Contributors">
        <img src="https://img.shields.io/github/contributors/Inria-Empenn/narps_open_pipelines" /></a>
    <a href="https://github.com/Inria-Empenn/narps_open_pipelines/pulse" alt="Commit activity">
        <img src="https://img.shields.io/github/commit-activity/m/Inria-Empenn/narps_open_pipelines" /></a>
</p>

## Table of contents

- [Project presentation](#project-presentation)
- [Getting Started](#getting-started)
	- [Contents overview](#contents-overview)
	- [Installation](#installation)
	- [Contributing](#contributing)
- [References](#references)
- [Funding](#funding)

## Project presentation

Neuroimaging workflows are highly flexible, leaving researchers with multiple possible options to analyze a dataset [(Carp, 2012)](https://www.frontiersin.org/articles/10.3389/fnins.2012.00149/full).
However, different analytical choices can cause variation in the results [(Botvinik-Nezer et al., 2020)](https://www.nature.com/articles/s41586-020-2314-9), leading to what was called a "vibration of effects" [(Ioannidis, 2008)](https://pubmed.ncbi.nlm.nih.gov/18633328/) also known as analytical variability. 

**The goal of the NARPS Open Pipelines project is to create a codebase reproducing the 70 pipelines of the NARPS project (Botvinik-Nezer et al., 2020) and share this as an open resource for the community**. 

To perform the reproduction, we are lucky to be able to use the [descriptions provided by the teams](https://github.com/poldrack/narps/blob/1.0.1/ImageAnalyses/metadata_files/analysis_pipelines_for_analysis.xlsx).
We also created a [shared spreadsheet](https://docs.google.com/spreadsheets/d/1FU_F6kdxOD4PRQDIHXGHS4zTi_jEVaUqY_Zwg0z6S64/edit?usp=sharing) that can be used to add comments on pipelines (e.g.: identify the ones that are not reproducible with NiPype).

:vertical_traffic_light: Lastly, please find [here in the project's wiki](https://github.com/Inria-Empenn/narps_open_pipelines/wiki/pipeline_status) a dashboard to see pipelines work progresses at first glance.

## Getting Started

### Contents overview

- :snake: :package: `narps_open/` contains the Python package with all the pipelines logic.
- :brain: `data/` contains data that is used by the pipelines, as well as the (intermediate or final) results data. Instructions to download data are available in [INSTALL.md](/INSTALL.md#data-download-instructions).
- :blue_book: `docs/` contains the documentation for the project. Start browsing it with the entry point [docs/README.md](/docs/README.md)
- :orange_book: `examples/` contains notebooks examples to launch of the reproduced pipelines.
- :microscope: `tests/` contains the tests of the narps_open package.

### Installation

To get the pipelines running, please follow the installation steps in [INSTALL.md](/INSTALL.md)

### Contributing 

:wave: Any help is welcome ! Follow the guidelines in [CONTRIBUTING.md](/CONTRIBUTING.md) if you wish to get involved !

## References

1. [Botvinik-Nezer, R. et al. (2020), ‘Variability in the analysis of a single neuroimaging dataset by many teams’, Nature.](https://www.nature.com/articles/s41586-020-2314-9)
2. [Carp, J. et al. (2012), ‘On the Plurality of (Methodological) Worlds: Estimating the Analytic Flexibility of fMRI Experiments’, Frontiers in Neuroscience.](https://www.frontiersin.org/articles/10.3389/fnins.2012.00149/full)
3. [Gorgolewski, K.J. et al. (2015), ‘NeuroVault.org: a web-based repository for collecting and sharing unthresholded statistical maps of the human brain’ Frontiers in Neuroinformatics.](https://www.frontiersin.org/articles/10.3389/fninf.2015.00008/full)
4. [Ioannidis, J.P.A. (2008), ‘Why Most Discovered True Associations Are Inflated’, Epidemiology.](https://pubmed.ncbi.nlm.nih.gov/18633328/)

## Funding

This project is supported by Région Bretagne (Boost MIND). 

## Credits

This project is developed in the Empenn team by Boris Clenet, Elodie Germani, Jeremy Lefort-Besnard and Camille Maumet with contributions by Rémi Gau.

In addition, this project was presented and received contributions during the following events:
 - OHBM Brainhack 2022: Elodie Germani, Arshitha Basavaraj, Trang Cao, Rémi Gau, Anna Menacher, Camille Maumet.
 - e-ReproNim FENS NENS Cluster Brainhack: <ADD_NAMES_HERE>
 - OHBM Brainhack 2023: <ADD_NAMES_HERE>
