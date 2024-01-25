# How to install NARPS Open Pipelines ? 

## 1 - Fork the repository

[Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the repository, so you have your own working copy of it.

## 2 - Clone the code

First, install [Datalad](https://handbook.datalad.org/en/latest/intro/installation.html#install-datalad). This will allow you to access the NARPS data easily, as it is included in the repository as [datalad subdatasets](http://handbook.datalad.org/en/latest/basics/101-106-nesting.html).

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

[Install Docker](https://docs.docker.com/engine/install/) then pull the nipype Docker image :

```bash
docker pull nipype/nipype
```

Once it's done you can check the image is available on your system :

```bash
docker images
   REPOSITORY                 TAG       IMAGE ID        CREATED         SIZE
   docker.io/nipype/nipype    latest    0f3c74d28406    9 months ago    22.7 GB
```

> [!NOTE]  
> Feel free to read this documentation page [docs/environment.md](docs/environment.md) to get further information about this environment.

## 5 - Run the project

Start a Docker container from the Docker image :

```bash
# Replace PATH_TO_THE_REPOSITORY in the following command (e.g.: with /home/user/dev/narps_open_pipelines/)
docker run -it -v PATH_TO_THE_REPOSITORY:/home/neuro/code/ nipype/nipype
```

Optionally edit the configuration file `narps_open/utils/default_config.toml` so that the referred paths match the ones inside the container. E.g.: if using the previous command line, the `directories` part of the configuration file should be :
```toml
# default_config.toml
# ...

[directories]
dataset = "/home/neuro/code/data/original/ds001734/"
reproduced_results = "/home/neuro/code/data/reproduced/"
narps_results = "/home/neuro/code/data/results/"

# ...
```

> [!NOTE]  
> Further information about configuration files can be found on the page [docs/configuration.md](docs/configuration.md).

Install NARPS Open Pipelines inside the container :

```bash
source activate neuro
cd /home/neuro/code/
pip install .
```

Finally, you are able to use the scripts of the project :

* `narps_open_runner`: run pipelines
* `narps_open_tester`: run a pipeline and test its results against original ones from the team
* `narps_description`: get the textual description made by a team
* `narps_results`: download the original results from teams
* `narps_open_status`: get status information about the development process of the pipelines

```bash
# Run the pipeline for team 2T6S, with 40 subjects
narps_open_runner -t 2T6S -n 40

# Run the pipeline for team 08MQ, compare results with original ones,
#   and produces a report with correlation values.
narps_open_tester -t 08MQ

# Get the description of team C88N in markdown formatting
narps_description -t C88N --md

# Download the results from all teams
narps_results -a

#  Get the pipeline work status information in json formatting
narps_open_status --json
```

> [!NOTE]  
> For further information about these command line tools, read the corresponding documentation pages.
> * `narps_open_runner` : [docs/running.md](docs/running.md)
> * `narps_open_tester` : [docs/testing.md](docs/testing.md#command-line-tool)
> * `narps_description` : [docs/description.md](docs/description.md)
> * `narps_results` : [docs/data.md](docs/data.md#results-from-narps-teams)
> * `narps_open_status` : [docs/status.md](docs/status.md)
