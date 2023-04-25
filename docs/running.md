# :running: How to run NARPS open pipelines ?

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

# Or start the first level only (preprocessing + run level + subject level)
runner.start(True, False)

# Or start the second level only (group level)
runner.start(True, True)
```

## Using the runner application

The `narps_open.runner` module also allows to run pipelines from the command line :

```bash
python narps_open/runner.py -h
	usage: runner.py [-h] -t TEAM (-r RANDOM | -s SUBJECTS [SUBJECTS ...]) [-g | -f]

	Run the pipelines from NARPS.

	options:
	  -h, --help            show this help message and exit
	  -t TEAM, --team TEAM  the team ID
	  -r RANDOM, --random RANDOM the number of subjects to be randomly selected
	  -s SUBJECTS [SUBJECTS ...], --subjects SUBJECTS [SUBJECTS ...] a list of subjects
	  -g, --group           run the group level only
	  -f, --first           run the first levels only (preprocessing + subjects + runs)

python narps_open/runner.py -t 2T6S -s 001 006 020 100
python narps_open/runner.py -t 2T6S -r 4
python narps_open/runner.py -t 2T6S -r 4 -f
```

In this usecase, the paths where to store the outputs and to the dataset are picked by the runner from the [configuration](docs/configuration.md).
