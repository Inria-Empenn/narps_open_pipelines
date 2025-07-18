# How to run NARPS open pipelines ? :running:

## Using the runner application

The `narps_open.runner` module allows to run pipelines from the command line.

> [!TIP]
> In the following examples, use `narps_open_runner` or `python narps_open/runner.py` indifferently to launch the command line tool.

```bash
narps_open_runner -h
	usage: narps_open_runner [-h] -t
	                         {08MQ,2T6S,3TR7,4SZ2,4TQ6,51PW,98BT,B23O,C88N,J7F9,L7J7,O21U,O6R6,Q6O0,R9K3,T54A,U26C,UK24,X19V}
	                         (-s SUBJECTS [SUBJECTS ...] | -n NSUBJECTS | -r RSUBJECTS) [-l {p,r,s,g} [{p,r,s,g} ...]]
	                         [-c] [-e]

	Run the pipelines from NARPS.

	options:
	  -h, --help            show this help message and exit
	  -t {08MQ,2T6S,3TR7,4SZ2,4TQ6,51PW,98BT,B23O,C88N,J7F9,L7J7,O21U,O6R6,Q6O0,R9K3,T54A,U26C,UK24,X19V}, --team {08MQ,2T6S,3TR7,4SZ2,4TQ6,51PW,98BT,B23O,C88N,J7F9,L7J7,O21U,O6R6,Q6O0,R9K3,T54A,U26C,UK24,X19V}
	                        the team ID
	  -s SUBJECTS [SUBJECTS ...], --subjects SUBJECTS [SUBJECTS ...]
	                        a list of subjects to be selected
	  -n NSUBJECTS, --nsubjects NSUBJECTS
	                        the number of subjects to be selected
	  -r RSUBJECTS, --rsubjects RSUBJECTS
	                        the number of subjects to be selected randomly
	  -l {p,r,s,g} [{p,r,s,g} ...], --levels {p,r,s,g} [{p,r,s,g} ...]
	                        the analysis levels to run (p=preprocessing, r=run, s=subject, g=group)
	  -c, --check           check pipeline outputs (runner is not launched)
	  -e, --exclusions      run the analyses without the excluded subjects
	  --config CONFIG       custom configuration file to be used

narps_open_runner -t 2T6S -s 001 006 020 100 # Launches the full pipeline on the given subjects
narps_open_runner -t 2T6S -r 4 # Launches the full pipeline on 4 random subjects
narps_open_runner -t 2T6S -r 4 -l s # Launches the subject level of the pipeline on 4 random subjects
narps_open_runner -t 2T6S -r 4 -l p r s -c # Check the output files of the prerprocessing, run level and subject level parts of the pipeline, without launching it.
```

> [!NOTE]
> In this usecase, the paths where to store the outputs and to the dataset are picked by the runner from the [configuration](docs/configuration.md).

## Using the `PipelineRunner` object

The class `PipelineRunner` is available from the `narps_open.runner` module. You can use it from inside python code, as follows :

```python
from narps_open.runner import PipelineRunner, PipelineRunnerLevel

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

# Start the runner (all available levels)
runner.start()

# Start the subject level only
runner.start(PipelineRunnerLevel.SUBJECT)

# Or start the "first level" (preprocessing + run level + subject level)
runner.start(PipelineRunnerLevel.FIRST)

# Or start the group level only
runner.start(PipelineRunnerLevel.GROUP)

# Get the list of missing files (if any) after the pipeline finished
runner.get_missing_outputs() # for all available levels
runner.get_missing_outputs(PipelineRunnerLevel.PREPROCESSING) # for preprocessing only
```
