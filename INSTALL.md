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

When the image is built, follow the instructions in [docs/environment.md](docs/environment.md) to start the environment from it.
