# How to run NARPS open pipelines ? :running:

## Using the runner application

The `narps_open.runner` module allows to run pipelines from the command line.

> [!TIP]
> In the following examples, use `narps_open_runner` or `python narps_open/runner.py` indifferently to launch the command line tool.

```bash
narps_open_runner -h
	usage: runner.py [-h] -t TEAM (-r RANDOM | -s SUBJECTS [SUBJECTS ...]) [-g | -f]

	Run the pipelines from NARPS.

	options:
	  -h, --help            show this help message and exit
	  -t TEAM, --team TEAM  the team ID
	  -r RANDOM, --random RANDOM the number of subjects to be randomly selected
	  -s SUBJECTS [SUBJECTS ...], --subjects SUBJECTS [SUBJECTS ...] a list of subjects
	  -g, --group           run the group level only
	  -f, --first           run the first levels only (preprocessing + subjects + runs)
	  -c, --check           check pipeline outputs (runner is not launched)

narps_open_runner -t 2T6S -s 001 006 020 100
narps_open_runner -t 2T6S -r 4
narps_open_runner -t 2T6S -r 4 -f
narps_open_runner -t 2T6S -r 4 -f -c # Check the output files without launching the runner
```

> [!NOTE]
> In this usecase, the paths where to store the outputs and to the dataset are picked by the runner from the [configuration](docs/configuration.md).

## Using the `PipelineRunner` object

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

# Get the list of missing files (if any) after the pipeline finished
runner.get_missing_first_level_outputs()
runner.get_missing_group_level_outputs()
```
