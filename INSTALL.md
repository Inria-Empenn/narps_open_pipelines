# How to install NARPS Open Pipelines ? 

## 1 - Fork the repository

[Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the repository, so you have your own working copy of it.

## 2 - Clone the code

First, install [Datalad](https://www.datalad.org/). This will allow you to access the NARPS data easily, as it is included in the repository as [datalad subdatasets](http://handbook.datalad.org/en/latest/basics/101-106-nesting.html).

Then, [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) the project :

```bash
# Replace YOUR_GITHUB_USERNAME in the following command.
datalad install --recursive https://github.com/YOUR_GITHUB_USERNAME/narps_open_pipelines.git
```

> [!WARNING]  
> It is still possible to clone the fork using [git](https://git-scm.com/) ; but by doing this, you will only get the code.
> ```bash
> # Replace YOUR_GITHUB_USERNAME in the following command.
> git clone https://github.com/YOUR_GITHUB_USERNAME/narps_open_pipelines.git
> ```

## 3 - Get the data

Now that you cloned the repository using Datalad, you are able to get the data :

```bash
# Move inside the root directory of the repository.
cd narps_open_pipelines

# Select the data you want to download. Here is an example to get data of the first 4 subjects.
datalad get data/original/ds001734/sub-00[1-4] -J 12
datalad get data/original/ds001734/derivatives/fmriprep/sub-00[1-4] -J 12
```

> [!NOTE]  
> For further information and alternatives on how to get the data, see the corresponding documentation page [docs/data.md](docs/data.md).

## 4 - Set up the environment

[Install Docker](https://docs.docker.com/engine/install/) then pull the Docker image :

```bash
docker pull elodiegermani/open_pipeline:latest
```

Once it's done you can check the image is available on your system :

```bash
docker images
   REPOSITORY                               TAG       IMAGE ID        CREATED         SIZE
   docker.io/elodiegermani/open_pipeline    latest    0f3c74d28406    9 months ago    22.7 GB
```

> [!NOTE]  
> Feel free to read this documentation page [docs/environment.md](docs/environment.md) to get further information about this environment.

## 5 - Run the project

Start a Docker container from the Docker image :

```bash
# Replace PATH_TO_THE_REPOSITORY in the following command (e.g.: with /home/user/dev/narps_open_pipelines/)
docker run -it -v PATH_TO_THE_REPOSITORY:/home/neuro/code/ elodiegermani/open_pipeline
```

Install NARPS Open Pipelines inside the container :

```bash
source activate neuro
cd /home/neuro/code/
pip install .
```

Finally, you are able to run pipelines :

```bash
python narps_open/runner.py
  usage: runner.py [-h] -t TEAM (-r RSUBJECTS | -s SUBJECTS [SUBJECTS ...] | -n NSUBJECTS) [-g | -f] [-c]
```

> [!NOTE]  
> For further information, read this documentation page [docs/running.md](docs/running.md).
