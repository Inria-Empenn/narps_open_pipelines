# THE OPEN PIPELINE PROJECT

<p align="center">
	<img src="https://github.com/elodiegermani/narps_open_pipelines/blob/main/static/images/project_illustration.png"/> 
</p>

## Table of contents
- [Table of contents](#table-of-contents)
- [Project presentation](#project-presentation)
- [Getting Started](#getting-started)
	- [Contents overview](#contents-overview)
		- [`src`](#src)
		- [`data`](#data)
	- [Install Docker container](#install-docker-container)
	- [Data Download Instructions](#data-download-instructions)
		- [Original Dataset](#original-dataset)
		- [Derived data](#derived-data)
	- [Contributing](#contributing)
- [References](#references)


## Project presentation

Neuroimaging workflows are highly flexible, leaving researchers with multiple possible options to analyze a dataset [(Carp, 2012)](https://www.frontiersin.org/articles/10.3389/fnins.2012.00149/full).
However, different analytical choices can cause variation in the results [(Botvinik-Nezer et al., 2020)](https://www.nature.com/articles/s41586-020-2314-9), leading to what was called a "vibration of effects" [(Ioannidis, 2008)](https://pubmed.ncbi.nlm.nih.gov/18633328/) also known as analytical variability. 

**The goal of the NARPS open pipeline project is to create a codebase reproducing the 70 pipelines of the NARPS project (Botvinik-Nezer et al., 2020) and share this as an open resource for the community**. 

To perform the reproduction, we are lucky to be able to use the description provided by each team available [here](https://github.com/poldrack/narps/blob/1.0.1/ImageAnalyses/metadata_files/analysis_pipelines_for_analysis.xlsx). 
We also created a shared spreadsheet that can be use to add comments on pipelines: the ones that are already reproduced, the ones that are not reproducible with NiPype... You can find it [here](https://docs.google.com/spreadsheets/d/1FU_F6kdxOD4PRQDIHXGHS4zTi_jEVaUqY_Zwg0z6S64/edit?usp=sharing).

## Getting Started

[Fork and clone](https://docs.github.com/en/get-started/quickstart/fork-a-repo) this repository to your local machine. 

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/narps_open_pipelines.git
```

If you are using [Datalad](#data-download-instructions),
because the NARPS data is "bundled" in this repository as 
[datalad subdataset](http://handbook.datalad.org/en/latest/basics/101-106-nesting.html), 
you may want to use datalad to clone your fork of the repository by doing:

```bash
datalad install --recursive https://github.com/YOUR_GITHUB_USERNAME/narps_open_pipelines.git
```


### Contents overview

#### `src`

This directory contains scripts of the reproduce pipelines along with notebooks and/or scripts to launch them. 

#### `data`

This directory is made to contain data that will be used by scripts/notebooks stored in the `src` directory and to contain the results (intermediate results and final data) of those scripts. 

Instructions to download data are available [below](#data-download-instructions).

### Install Docker container

To use the notebooks and launch the pipelines, you need to install the [NiPype](https://nipype.readthedocs.io/en/latest/users/install.html) Python package but also the original software package used in the pipeline (SPM, FSL, AFNI...). 

To facilitate this step, we created a Docker container based on [Neurodocker](https://github.com/ReproNim/neurodocker) that contains the necessary Python packages and software packages. To install the Docker image, two options are available.

#### Option 1: Using Dockerhub
```bash
docker pull elodiegermani/open_pipeline:latest
```

The image should install itself. Once it's done you can check available images on your system:

```bash
docker images
```

#### Option 2: Using a Dockerfile 
The Dockerfile used for the image stored on Dockerhub is available on the GitHub repository. But you might want to personalize your Dockerfile to install only the necessary software packages. To do so, modify the command below to modify the Dockerfile: 

```bash
docker run --rm repronim/neurodocker:0.7.0 generate docker \
           --base neurodebian:stretch-non-free --pkg-manager apt \
           --install git \
           --fsl version=6.0.3 \
           --afni version=latest method=binaries install_r=true install_r_pkgs=true install_python2=true install_python3=true \
           --spm12 version=r7771 method=binaries \
           --user=neuro \
           --workdir /home \
           --miniconda create_env=neuro \
                       conda_install="python=3.8 traits jupyter nilearn graphviz nipype scikit-image" \
                       pip_install="matplotlib" \
                       activate=True \
           --env LD_LIBRARY_PATH="/opt/miniconda-latest/envs/neuro:$LD_LIBRARY_PATH" \
           --run-bash "source activate neuro" \
           --user=root \
           --run 'chmod 777 -Rf /home' \
           --run 'chown -R neuro /home' \
           --user=neuro \
           --run 'mkdir -p ~/.jupyter && echo c.NotebookApp.ip = \"0.0.0.0\" > ~/.jupyter/jupyter_notebook_config.py' > Dockerfile
```

When you are satisfied with your Dockerfile, just build the image:

```bash
docker build --tag [name_of_the_image] - < Dockerfile
```

When the installation is finished, you have to build a container using the command below:

```bash
docker run 	-ti \
		-p 8888:8888 \
		elodiegermani/open_pipeline
```

On this command line, you need to add volumes to be able to link with your local files (original dataset and git repository). If you stored the original dataset in `data/original`, just make a volume with the `narps_open_pipelines` directory:

```bash
docker run 	-ti \
		-p 8888:8888 \
		-v /users/egermani/Documents/narps_open_pipelines:/home/ \
		elodiegermani/open_pipeline
``` 

If it is in another directory, make a second volume with the path to your dataset:

```bash
docker run 	-ti \
		-p 8888:8888 \
		-v /Users/egermani/Documents/narps_open_pipelines:/home/ \
		-v /Users/egermani/Documents/data/NARPS/:/data/ \
		elodiegermani/open_pipeline
```

After that, your container will be launched! 

#### Other command that could be useful: 
##### START THE CONTAINER 

```bash
docker start [name_of_the_container]
```

##### VERIFY THE CONTAINER IS IN THE LIST 

```bash
docker ps
```

##### EXECUTE BASH OR ATTACH YOUR CONTAINER 

```bash
docker exec -ti [name_of_the_container] bash
```

**OR**

```bash
docker attach [name_of_the_container]
```

#### Useful command inside the container: 
##### ACTIVATE CONDA ENVIRONMENT

```bash
source activate neuro
```

##### LAUNCH JUPYTER NOTEBOOK

```bash
jupyter notebook --port=8888 --no-browser --ip=0.0.0.0
```

#### If you did not use your container for a while: 
##### VERIFY IT STILL RUN : 

```bash
docker ps -l
```

##### IF YOUR DOCKER CONTAINER IS IN THE LIST, RUN :

```bash
docker start [name_of_the_container]
```

##### ELSE, RERUN IT WITH : 

```bash
docker run 	-ti \
		-p 8888:8888 \
		-v /home/egermani:/home \
		[name_of_the_image]
```

#### To use SPM inside the container, use this command at the beginning of your script:

```python
from nipype.interfaces import spm

matlab_cmd = '/opt/spm12-r7771/run_spm12.sh /opt/matlabmcr-2010a/v713/ script'

spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)
```

### Data Download Instructions

The dataset used for the `/src/reproduction_*.ipynb` can be downloaded using one of the three options below. 

For the scripts to work as intended, the dataset **MUST** be stored in `data/original`.

On your local copy, place yourself in `data/original` directory before running the commands below.
#### Original Dataset

**Option 1: with DataLad (Recommended)**

1. Install DataLad and it's dependencies from **[here](http://handbook.datalad.org/en/latest/intro/installation.html)**, if you don't have it installed already.

Tips for people using M1 MacBooks: git-annex is not yet available for M1 MacBooks. A solution to install it can be found [here](https://gist.github.com/Arshitha/45026e56b71ae35446af2239f98dcb4b). 
   
2. If you have [cloned this repository using Datalad](#getting-started) with the `--recusive` option
   then the dataset should mostly already be in `data/original/ds001734`

The `datalad install` command only downloads the metadata associated with the dataset. 
To download the actual files run `datalad get ./*` and if you only want parts of the data, 
replace the `*` by the paths to the desired files.

**Option 2: with Node.js**

Using [@openneuro/cli](https://www.npmjs.com/package/@openneuro/cli) you can download this dataset from the command line using [Node.js](https://nodejs.org/en/download/). This method is good for larger datasets or unstable connections, but has known issues on Windows.

```bash
openneuro download --snapshot 1.0.5 ds001734 ds001734/
```

This will download to `ds001734/` in the current directory. If your download is interrupted and you need to retry, rerun the command to resume the download.

**Option 3: from S3**

The most recently published snapshot can be downloaded from S3. This method is best for larger datasets or unstable connections. This example uses [AWS CLI](https://aws.amazon.com/cli/).

```bash
aws s3 sync --no-sign-request s3://openneuro.org/ds001734 ds001734-download/
```

File containing pipeline description is available in `/data/original`.

Download Instructions Source: https://openneuro.org/datasets/ds001734/versions/1.0.5/download
#### Derived data

Derived data such as original stat maps from teams and reproduced stat maps 
can be downloaded from [NeuroVault](https://www.neurovault.org) 
[(Gorgolewski & al, 2015)](https://www.frontiersin.org/articles/10.3389/fninf.2015.00008/full). 

The original derivative data from the NARPS paper are available 
as release on zenodo: https://zenodo.org/record/3528329/

There are aslo included as a datalad subdataset in `data/neurovault`.

Each teams results is kept in the `orig` in folder organized using the pattern:

```
neurovaultCollectionNumber_teamID
```

### Contributing 

Follow the guidelines in [CONTRIBUTING.md](https://github.com/elodiegermani/open_pipeline/blob/main/CONTRIBUTING.md)

## References

1. [Botvinik-Nezer, R. et al. (2020), ???Variability in the analysis of a single neuroimaging dataset by many teams???, Nature.](https://www.nature.com/articles/s41586-020-2314-9)
2. [Carp, J. et al. (2012), ???On the Plurality of (Methodological) Worlds: Estimating the Analytic Flexibility of fMRI Experiments???, Frontiers in Neuroscience.](https://www.frontiersin.org/articles/10.3389/fnins.2012.00149/full)
3. [Gorgolewski, K.J. et al. (2015), ???NeuroVault.org: a web-based repository for collecting and sharing unthresholded statistical maps of the human brain??? Frontiers in Neuroinformatics.](https://www.frontiersin.org/articles/10.3389/fninf.2015.00008/full)
4. [Ioannidis, J.P.A. (2008), ???Why Most Discovered True Associations Are Inflated???, Epidemiology.](https://pubmed.ncbi.nlm.nih.gov/18633328/)
