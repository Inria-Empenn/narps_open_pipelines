# Handle data :brain: for the NARPS open pipelines project

The datasets used for the project can be downloaded using one of the two options below.

The path to these datasets must conform with the information located in the configuration file you plan to use (cf. [documentation about configuration](docs/configuration.md)). By default, these paths are in the repository:
   * `data/original/`: original data from NARPS
   * `data/results/`: results from NARPS teams

In the following, we assume you choose this configuration.

## Option 1: with DataLad (recommended)

1. If you don't have it installed already, install DataLad and it's dependencies from [here](http://handbook.datalad.org/en/latest/intro/installation.html).

Tips for people using M1 MacBooks: `git-annex` is not yet available for M1 MacBooks. A solution to install it can be found [here](https://gist.github.com/Arshitha/45026e56b71ae35446af2239f98dcb4b). 

2. If you have cloned the Narps Open repository using DataLad with the `datalad install --recusive` option, then the datasets are already available in:
   * `data/original/`: original data from NARPS
   * `data/results/`: results from NARPS teams

The `datalad install` command only downloaded the metadata associated with the dataset ; to download the actual files run the following command:

```bash
# To get all the data
cd data/
datalad get ./*
```

If you only want parts of the data, replace the `./*` by the paths to the desired files.

## Option 2: manual download

### Original data from NARPS

Follow the download instructions from openneuro.org for the ds001734 dataset [here](https://openneuro.org/datasets/ds001734/versions/1.0.5/download). You will end up typing one of these commands:

```bash
cd data/original/
openneuro download --snapshot 1.0.5 ds001734 ds001734/

# or

cd data/original/
aws s3 sync --no-sign-request s3://openneuro.org/ds001734 ds001734/
```

### Results from NARPS teams

Stat maps from teams can be downloaded from [NeuroVault](https://www.neurovault.org) [(Gorgolewski & al, 2015)](https://www.frontiersin.org/articles/10.3389/fninf.2015.00008/full).

The `narps_open.data.results` module will help you download these collections. Note that it is also possible to rectify the collection, i.e: to pre-process the images as done by the NARPS analysis team during the NARPS study.

Here is how to use the module, both using python code or with the command line:

```python
# In a python script
from narps_open.data.results import ResultsCollectionFactory

# Create a collection factory
factory = ResultsCollectionFactory()

# Select the collections you need
teams = ['2T6S', 'C88N', 'L1A8'] # Alternatively use the keys from narps_open.pipelines.implemented_pipelines to get all the team ids
for team in teams:
    collection = factory.get_collection(team)
    collection.download() # Collections are downloaded
    collection.rectify() # Rectified versions are created
```

```bash
# From the command line
$ python narps_open/data/results -h
usage: results [-h] (-t TEAMS [TEAMS ...] | -a) [-r]

Get Neurovault collection of results from NARPS teams.

options:
  -h, --help            show this help message and exit
  -t TEAMS [TEAMS ...], --teams TEAMS [TEAMS ...]
                        a list of team IDs
  -a, --all             download results from all teams
  -r, --rectify         rectify the results

# Either download all collections
python narps_open/utils/results -a

# Or select the ones you need
python narps_open/utils/results -t 2T6S C88N L1A8

# Download and rectify the collections
python narps_open/utils/results -r -t 2T6S C88N L1A8
```

The collections are also available [here](https://zenodo.org/record/3528329/) as one release on Zenodo that you can download.

Each team results collection is kept in the `data/results/orig` directory, in a folder using the pattern `<neurovault_collection_id>_<team_id>` (e.g.: `4881_2T6S` for the 2T6S team).
