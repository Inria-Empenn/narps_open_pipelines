# Utils to tests pipelines

from os.path import join

from narps_open.runner import PipelineRunner
from narps_open.utils.correlation import get_correlation_coefficient

def test_pipeline(team_id: str, dataset_dir: str, results_dir: str, nb_subjects: int = 4):
	""" Iterate over subject number """

	# Initialize pipeline
	runner = PipelineRunner(team_id)
    runner.random_nb_subjects = nb_subjects
	runner.pipeline.directories.dataset_dir = dataset_dir
    runner.pipeline.directories.results_dir = results_dir
    runner.pipeline.directories.set_output_dir_with_team_id(team_id)
    runner.pipeline.directories.set_working_dir_with_team_id(team_id)
    runner.start()

    # Where are the files ?
    join(runner.pipeline.directories.output_dir, 'datasink', 'con00001')

    # Compute correlation coefficient
    get_correlation_coefficient('tmp_image_1.nii', 'tmp_image_1.nii')

### This fucntion can be used as follows:
assert test_pipeline('2T6S', '/data/', '/output/', 4) > .5
assert test_pipeline('2T6S', '/data/', '/output/', 8) > .6
assert test_pipeline('2T6S', '/data/', '/output/', 10) > .7
assert test_pipeline('2T6S', '/data/', '/output/', 50) > .8
assert test_pipeline('2T6S', '/data/', '/output/', 100) > .9

### TODO : how to keep intermediate files of the low level for the next numbers of subjects ?
### TODO : where is the reference / gold truth dataset ?
