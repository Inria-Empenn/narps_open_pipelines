# How to install NARPS Open Pipelines ? 

## 1 - Get the code

First, [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the repository, so you have your own working copy of it.

Then, you have two options to [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) the project :

### Option 1: Using DataLad (recommended)

Cloning the fork using [Datalad](https://www.datalad.org/) will allow you to get the code as well as "links" to the data, because the NARPS data is bundled in this repository as [datalad subdatasets](http://handbook.datalad.org/en/latest/basics/101-106-nesting.html).

```bash
datalad install --recursive https://github.com/YOUR_GITHUB_USERNAME/narps_open_pipelines.git
```

### Option 2: Using Git

Cloning the fork using [git](https://git-scm.com/) ; by doing this, you will only get the code.

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/narps_open_pipelines.git
```

## 2 - Get the data

Ignore this step if you used DataLad (option 1) in the previous step.

Otherwise, there are several ways to get the data.

## 3 - Set up the environment

The Narps Open Pipelines project is build upon several dependencies, such as [Nipype](https://nipype.readthedocs.io/en/latest/) but also the original software packages used by the pipelines (SPM, FSL, AFNI...). 

To facilitate this step, we created a Docker container based on [Neurodocker](https://github.com/ReproNim/neurodocker) that contains the necessary Python packages and software. To install the Docker image, two options are available.

### Option 1: Using Dockerhub

```bash
docker pull elodiegermani/open_pipeline:latest
```

The image should install itself. Once it's done you can check the image is available on your system:

```bash
docker images
   docker.io/elodiegermani/open_pipeline    latest    0f3c74d28406    9 months ago    22.7 GB
```

### Option 2: Using a Dockerfile 

The Dockerfile used to create the image stored on DockerHub is available at the root of the repository ([Dockerfile](Dockerfile)). But you might want to personalize this Dockerfile. To do so, change the command below that will generate a new Dockerfile: 

```bash
	docker run --rm repronim/neurodocker:0.9.5 generate docker \
			--base-image centos:7 --pkg-manager yum \
			--yes \
			--install git \
			--ants method=binaries version=2.4.3 \
			--fsl version=6.0.6.4 \
			--spm12 version=r7771 method=binaries \
			--miniconda method=binaries \
						version=latest \
						conda_install="python=3.10 pip=23.2.1" \
						pip_install="traits==6.3.0 jupyterlab-4.0.6 graphviz-0.20.1 nipype==1.8.6 scikit-image==0.21.0 matplotlib==3.8.0 nilearn==0.10.1" \
			--run 'mkdir -p ~/.jupyter && echo c.NotebookApp.ip = \"0.0.0.0\" > ~/.jupyter/jupyter_notebook_config.py' > Dockerfile
```

When you are satisfied with your Dockerfile, just build the image:

```bash
docker build --tag [name_of_the_image] - < Dockerfile
```

When the image is built, follow the instructions in [docs/environment.md](docs/environment.md) to start the environment from it.
