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

To perform the reproduction, we are lucky to be able to use the description provided by each team available [here](https://github.com/poldrack/narps/blob/1.0.1/ImageAnalyses/metadata_files/analysis_pipelines_for_analysis.xlsx).

## To start 

### Contents overview

#### `src`

This directory contains scripts of the reproduce pipelines along with notebooks and/or scripts to launch them. 

#### `data`

This directory is made to contain data that will be used by scripts/notebooks stored in the `src` directory and to contain the results (intermediate results and final data) of those scripts. 

Instructions to download data are available [below](#download-data).

### Install Docker container

To use the notebooks and launch the pipelines, you need to install the [NiPype](https://nipype.readthedocs.io/en/latest/users/install.html) Python package but also the original software package used in the pipeline (SPM, FSL, AFNI...). 

To facilitate this step, we created a Docker container based on [Neurodocker](https://github.com/ReproNim/neurodocker) that contains the necessary Python packages and software packages. To install the Docker image, use the command below: 
```
docker pull elodiegermani/open_pipeline:latest
```

The image should install itself. Once it's done you can check available images on your system:
```
docker images
```

When the installation is finished, you have to build a container using the command below:
```
docker run -ti -p 8888:8888 elodiegermani/open_pipeline
```

On this commandline, you need to add volumes to be able to link with your local files (original dataset and git repository). If you stored the original dataset in `data/original`, just make a volume with the `narps_open_pipelines` directory:
```
docker run -ti -p 8888:8888 -v /users/egermani/Documents/narps_open_pipelines:/home/ elodiegermani/open_pipeline
``` 

If it is in another directory, make a second volume with the path to your dataset:
```
docker run -ti -p 8888:8888 -v /Users/egermani/Documents/narps_open_pipelines:/home/ -v /Users/egermani/Documents/data/NARPS/:/data/ elodiegermani/open_pipeline
```

After that, your container will be launched! 

#### Other command that could be useful: 
##### START THE CONTAINER 
```docker start [name_of_the_container]```

##### VERIFY THE CONTAINER IS IN THE LIST 
```docker ps ```

##### EXECUTE BASH OR ATTACH YOUR CONTAINER 
```docker exec -ti [name_of_the_container] bash```
OR
```docker attach [name_of_the_container]```

#### Useful command inside the container: 
##### ACTIVATE CONDA ENVIRONMENT
```source activate neuro```

##### LAUNCH JUPYTER NOTEBOOK
```jupyter notebook --port=8888 --no-browser --ip=0.0.0.0```

#### If you did not use your container for a while: 
##### VERIFY IT STILL RUN : 
```docker ps -l```
##### IF YOUR DOCKER CONTAINER IS IN THE LIST, RUN : 
```docker start [name_of_the_container]```
##### ELSE, RERUN IT WITH : 
```docker run -ti -p 8888:8888 -v /home/egermani:/home [name_of_the_image]```

#### To use SPM inside the container, use this command at the beginning of your script:
```
from nipype.interfaces import spm
matlab_cmd = '/opt/spm12-r7771/run_spm12.sh /opt/matlabmcr-2010a/v713/ script'
spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)
```

### Download data 

#### Original dataset

The dataset used for the `/src/reproduction_*.ipynb` notebooks can be downloaded [**here**](https://openneuro.org/datasets/ds001734/versions/1.0.5).

I recommand to store the data in a directory inside the `data/original` directory. 

I recommand to download it with **Datalad**. If you want to use it, install [**Datalad**](http://handbook.datalad.org/en/latest/intro/installation.html#install), place yourself in the `data/original` directory and run `datalad install ///openneuro/ds001734`.
After, you can download all the files by running `datalad get ./*` and if you only want parts of the data, replace the * by the paths to the desired files. 

#### Derived data

Derived data such as original stat maps from teams and reproduced stat maps can be downloaded from [NeuroVault](https://www.neurovault.org) (Gorgolewski & al, 2015). 

*Coming soon*

### Contributing 

Follow the guidelines in [CONTRIBUTING.md](https://github.com/elodiegermani/open_pipeline/blob/main/CONTRIBUTING.md)

## References

- Botvinik-Nezer, R. et al. (2020), ‘Variability in the analysis of a single neuroimaging dataset by many teams’, Nature.
- Carp, J. et al. (2012), ‘On the Plurality of (Methodological) Worlds: Estimating the Analytic Flexibility of fMRI Experiments’, Frontiers in Neuroscience. ;
- Gorgolewski, K.J. et al. (2015), ‘NeuroVault.org: a web-based repository for collecting and sharing unthresholded statistical maps of the human brain’ Frontiers in Neuroinformatics. ;
- Ioannidis, J.P.A. (2008), ‘Why Most Discovered True Associations Are Inflated’, Epidemiology. ;
