# How to run NARPS open pipelines ? :running:

## Using the `PipelineRunner`

The class `PipelineRunner` is available from the `narps_open.runner` module. You can use it from inside python code, as follows :

```python
from narps_open.runner import PipelineRunner

# Initialize a PipelineRunner by choosing the team ID
runner = PipelineRunner(team_id = '2T6S')

# Set input and output directories
runner.pipeline.directories.dataset_dir = '/data/ds001734/'
runner.pipeline.directories.results_dir = '/output/'
runner.pipeline.directories.set_output_dir_with_team_id(runner.team_id)
runner.pipeline.directories.set_working_dir_with_team_id(runner.team_id)

# Set participants / subjects
runner.subjects = ['001', '006', '020', '100']

# Alternatively, ask the runner to pick a random number of subjects
# runner.random_nb_subjects = 4

# Start the runner
runner.start()
```

## Using the runner application

The `narps_open.runner` module also allows to run pipelines from the command line :

```bash
python narps_open/runner.py
	usage: runner.py [-h] -t TEAM -d DATASET -o OUTPUT (-r RANDOM | -s SUBJECTS [SUBJECTS ...])

python narps_open/runner.py -t 2T6S -d /data/ds001734/ -o /output/ -s 001 006 020 100
python narps_open/runner.py -t 2T6S -d /data/ds001734/ -o /output/ -r 4
```

* `-t` lets you set the team ID
* `-d` lets you set the dataset directory
* `-o` lets you set the output directory
* `-s` lets you set the list of subjects, alternatively, `-r` for a random number of subjects
