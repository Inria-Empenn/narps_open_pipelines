# Handle data :brain: for the NARPS open pipelines project

The datasets used for the project can be downloaded using one of the two options below.

The path to these datasets must conform with the information located in the configuration file you plan to use (cf. [documentation about configuration](/docs/configuration.md)). By default, these paths are in the repository:
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

> [!TIP]
> In the following examples, use `narps_results` or `python narps_open/data/results` indifferently to launch the command line tool.

```bash
# From the command line
narps_results -h
    usage: results [-h] (-t TEAMS [TEAMS ...] | -a) [-r]

    Get Neurovault collection of results from NARPS teams.

    options:
      -h, --help            show this help message and exit
      -t TEAMS [TEAMS ...], --teams TEAMS [TEAMS ...]
                            a list of team IDs
      -a, --all             download results from all teams
      -r, --rectify         rectify the results

# Either download all collections
narps_results -a

# Or select the ones you need
narps_results -t 2T6S C88N L1A8

# Download and rectify the collections
narps_results -r -t 2T6S C88N L1A8
```

The collections are also available [here](https://zenodo.org/record/3528329/) as one release on Zenodo that you can download.

Each team results collection is kept in the `data/results/orig` directory, in a folder using the pattern `<neurovault_collection_id>_<team_id>` (e.g.: `4881_2T6S` for the 2T6S team).

## Access NARPS data

Inside `narps_open.data`, several modules allow to parse data from the NARPS file, so it's easier to use it inside the Narps Open Pipelines project. These are :

### `narps_open.data.description`
Get textual description of the pipelines, as written by the teams (see [docs/description.md](/docs/description.md)).

### `narps_open.data.results`
Get the result collections, as described earlier in this file.

### `narps_open.data.participants`
Get the participants data (parses the `data/original/ds001734/participants.tsv` file) as well as participants subsets to perform analyses on lower numbers of images.

### `narps_open.data.task`
Get information about the task (parses the `data/original/ds001734/task-MGT_bold.json` file). Here is an example how to use it :

```python
from narps_open.data.task import TaskInformation

task_info = TaskInformation() # task_info is a dict

# All available keys
print(task_info.keys())
# dict_keys(['TaskName', 'Manufacturer', 'ManufacturersModelName', 'MagneticFieldStrength', 'RepetitionTime', 'EchoTime', 'FlipAngle', 'MultibandAccelerationFactor', 'EffectiveEchoSpacing', 'SliceTiming', 'BandwidthPerPixelPhaseEncode', 'PhaseEncodingDirection', 'TaskDescription', 'CogAtlasID', 'NumberOfSlices', 'AcquisitionTime', 'TotalReadoutTime'])

# Original data
print(task_info['TaskName'])
print(task_info['Manufacturer'])
print(task_info['RepetitionTime']) # And so on ...

# Derived data
print(task_info['NumberOfSlices'])
print(task_info['AcquisitionTime'])
print(task_info['TotalReadoutTime'])
```
