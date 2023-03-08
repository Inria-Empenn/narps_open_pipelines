# Utils to tests pipelines

from os.path import join

from narps_open.runner import PipelineRunner
from narps_open.utils.correlation import get_correlation_coefficient

def test_pipeline(team_id: str, dataset_dir: str, results_dir: str, nb_subjects: int = 4):
    """ This function allows to launch a pipeline over a given number of subjects

    Arguments:
        - team_id: str, the ID of the team (allows to identify which pipeline to run)
        - dataset_dir: str, the path to the ds001734 dataset
        - results_dir: str, the path where to store the results
        - nb_subjects: int, the number of subject to run the pipeline with

    Returns:
        - list(float) the correlation coefficients between the following
        (reference and computed) files:
    """

    # Initialize the pipeline
    runner = PipelineRunner(team_id)
    runner.random_nb_subjects = nb_subjects
    runner.pipeline.directories.dataset_dir = dataset_dir
    runner.pipeline.directories.results_dir = results_dir
    runner.pipeline.directories.set_output_dir_with_team_id(team_id)
    runner.pipeline.directories.set_working_dir_with_team_id(team_id)
    runner.start()

    # Build the path to the computed files
    out_path = join(runner.pipeline.directories.output_dir, 'datasink', 'con00001')

    # Compute the correlation coefficients
    return [
        get_correlation_coefficient('tmp_image_1.nii', 'tmp_image_1.nii'),
        get_correlation_coefficient('tmp_image_1.nii', 'tmp_image_1.nii')
        ]


# Example paths for reference data 2T6S
'https://neurovault.org/collections/4881/NARPS-2T6S/hypo1_unthresh'
'https://neurovault.org/collections/4881/NARPS-2T6S/hypo2_unthresh'
'https://neurovault.org/collections/4881/NARPS-2T6S/hypo3_unthresh'
'https://neurovault.org/collections/4881/NARPS-2T6S/hypo4_unthresh'
'https://neurovault.org/collections/4881/NARPS-2T6S/hypo5_unthresh'
'https://neurovault.org/collections/4881/NARPS-2T6S/hypo6_unthresh'
'https://neurovault.org/collections/4881/NARPS-2T6S/hypo7_unthresh'
'https://neurovault.org/collections/4881/NARPS-2T6S/hypo8_unthresh'
'https://neurovault.org/collections/4881/NARPS-2T6S/hypo9_unthresh'

# Need a conversion between pipeline's output data and reference data
#  > each pipeline must implement a convertor or always store data the same way



### This function can be used as follows:
assert test_pipeline('2T6S', '/data/', '/output/', 4) > .5
assert test_pipeline('2T6S', '/data/', '/output/', 8) > .6
assert test_pipeline('2T6S', '/data/', '/output/', 10) > .7
assert test_pipeline('2T6S', '/data/', '/output/', 50) > .8
assert test_pipeline('2T6S', '/data/', '/output/', 108) > .9

### TODO : how to keep intermediate files of the low level for the next numbers of subjects ?
#    - keep intermediate levels : boolean in PipelineRunner
### TODO : where is the reference / gold truth dataset ?
