# About the environment of NARPS Open Pipelines

## The Docker container :whale:

The NARPS Open Pipelines project is build upon several dependencies, such as [Nipype](https://nipype.readthedocs.io/en/latest/) but also the original software packages used by the pipelines (SPM, FSL, AFNI...). Therefore, we created a Docker container based on [Neurodocker](https://github.com/ReproNim/neurodocker) that contains software dependencies.

The simplest way to start the container using the command below :

```bash
docker run -it elodiegermani/open_pipeline
```

From this command line, you need to add volumes to be able to link with your local files (code repository).

```bash
# Replace PATH_TO_THE_REPOSITORY in the following command (e.g.: with /home/user/dev/narps_open_pipelines/)
docker run -it \
           -v PATH_TO_THE_REPOSITORY:/home/neuro/code/ \
           elodiegermani/open_pipeline
``` 

## Use Jupyter with the container

If you wish to use [Jupyter](https://jupyter.org/) to run the code, a port forwarding is needed :

```bash
docker run -it \
           -v PATH_TO_THE_REPOSITORY:/home/neuro/code/ \
           -p 8888:8888 \
           elodiegermani/open_pipeline
``` 

Then, from inside the container :

```bash
jupyter notebook --port=8888 --no-browser --ip=0.0.0.0
```

You can now access Jupyter using the address provided by the command line.

> [!NOTE]  
> Find useful information on the [Docker documentation page](https://docs.docker.com/get-started/). Here is a [cheat sheet with Docker commands](https://docs.docker.com/get-started/docker_cheatsheet.pdf)

## Create a custom Docker image

The `elodiegermani/open_pipeline` Docker image is based on [Neurodocker](https://github.com/ReproNim/neurodocker). It was created using the following command line :

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

If you wish to create your own custom environment, make changes to the parameters, and build your custom image from the generated Dockerfile.

```bash
# Replace IMAGE_NAME in the following command
docker build --tag IMAGE_NAME - < Dockerfile
```

## Good to know

To use SPM inside the container, use this command at the beginning of your script:

```python
from nipype.interfaces import spm

matlab_cmd = '/opt/spm12-r7771/run_spm12.sh /opt/matlabmcr-2010a/v713/ script'

spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)
```
