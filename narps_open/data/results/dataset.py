#!/usr/bin/python
# coding: utf-8

""" Generate a bash script to create the NARPS results dataset
    Warning: unfortunately, the script actually downloads the data.
"""

from os.path import join
from argparse import ArgumentParser

from narps_open.utils.configuration import Configuration
from narps_open.data.results import ResultsCollectionFactory
from narps_open.pipelines import implemented_pipelines

if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser(
        description='Generate a bash script to create the NARPS results dataset.'
        )
    parser.add_argument('-r', '--repository', type=str, required=True,
        help='adress of the repository where to push the dataset'
        )
    arguments = parser.parse_args()

    # Handle dataset directory
    dataset_dir = Configuration()['directories']['narps_results']

    # Create a new dataset
    print(f'mkdir -p {dataset_dir}')
    print(f'datalad create -D "NARPS results dataset" {dataset_dir}')

    # Add files for each team results collection
    collection_factory = ResultsCollectionFactory()
    for team_id, _ in implemented_pipelines.items():

        # Init collection
        collection = collection_factory.get_collection(team_id)

        # Create download directory if not existing
        print(f'mkdir -p {collection.directory}')

        # Create dataset entries
        for file_name, file_url in collection.files.items():
            complete_file_name = join(collection.directory, file_name+".nii.gz")
            short_file_name = complete_file_name.replace(dataset_dir, '')
            command = 'datalad download-url'
            command += f' -m \"New file {short_file_name}\"'
            command += f' --path \"{complete_file_name}\"'
            command += f' --dataset \"{dataset_dir}\"'
            command += f' \"{file_url}\"'
            print(command)

    # Push dataset
    print(f'cd {dataset_dir}')
    print(f'git remote add origin {arguments.repository}')
    print('git push -u origin master')
    print('datalad push')
