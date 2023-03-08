# Utils to tests pipelines

from os.path import join
from statistics import mean

from narps_open.runner import PipelineRunner
from narps_open.utils.correlation import get_correlation_coefficient

def test_pipeline(
    team_id: str,
    references_dir: str,
    dataset_dir: str,
    results_dir: str,
    nb_subjects: int = 4
    ):
    """ This function allows to launch a pipeline over a given number of subjects

    Arguments:
        - team_id: str, the ID of the team (allows to identify which pipeline to run)
        - references_dir: str, the path to the directory where results from the teams are
        - dataset_dir: str, the path to the ds001734 dataset
        - results_dir: str, the path where to store the results
        - nb_subjects: int, the number of subject to run the pipeline with

    Returns:
        - list(float) the correlation coefficients between the following
        (reference and computed) files:
    """

    # Initialize the pipeline
    """
    runner = PipelineRunner(team_id)
    runner.random_nb_subjects = nb_subjects
    runner.pipeline.directories.dataset_dir = dataset_dir
    runner.pipeline.directories.results_dir = results_dir
    runner.pipeline.directories.set_output_dir_with_team_id(team_id)
    runner.pipeline.directories.set_working_dir_with_team_id(team_id)
    runner.start()
    """

    # Retrieve the paths to the computed files
    output_files = [
        join(
            #runner.pipeline.directories.output_dir,
            '/home/bclenet/output',
            'NARPS-reproduction',
            f'team_2T6S_nsub_{nb_subjects}_hypo{hypothesis}_unthresholded.nii')
            #f'team-2T6S_nsub-{nb_subjects}_hypo-{hypothesis}_unthresholded.nii')
        for hypothesis in range(1, 10)
        ]

    # Retrieve the paths to the reference files
    reference_files = [
        join(
            references_dir,
            f'NARPS-{team_id}',
            f'hypo{hypothesis}_unthresholded.nii.gz')
        for hypothesis in range(1, 10)
        ]

    # Example paths for reference data 2T6S
    #    'https://neurovault.org/collections/4881/NARPS-2T6S/hypo1_unthresh'

    # Compute the correlation coefficients
    table = zip(output_files, reference_files)
    table = list(table)[1:4]
    return [
        get_correlation_coefficient(output_file, reference_file)
        for output_file, reference_file in table
        ]

### This function can be used as follows:

print(mean(test_pipeline('2T6S', '/home/bclenet/data/narps_nv_collections/', '/data/', '/output/', 4)))
print(test_pipeline('2T6S', '/home/bclenet/data/narps_nv_collections/', '/data/', '/output/', 4))

"""
assert mean(test_pipeline('2T6S', '/home/bclenet/data/narps_nv_collections/', '/data/', '/output/', 4)) > .5
assert mean(test_pipeline('2T6S', '/home/bclenet/data/narps_nv_collections/', '/data/', '/output/', 8)) > .6
assert mean(test_pipeline('2T6S', '/home/bclenet/data/narps_nv_collections/', '/data/', '/output/', 10)) > .7
assert mean(test_pipeline('2T6S', '/home/bclenet/data/narps_nv_collections/', '/data/', '/output/', 50)) > .8
assert mean(test_pipeline('2T6S', '/home/bclenet/data/narps_nv_collections/', '/data/', '/output/', 108)) > .9
"""
### TODO : how to keep intermediate files of the low level for the next numbers of subjects ?
#    - keep intermediate levels : boolean in PipelineRunner
### TODO : where is the reference / gold truth dataset ?
