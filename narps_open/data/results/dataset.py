#!/usr/bin/python
# coding: utf-8

""" Create the results dataset for the NARPS Open Pipelines project """

from os.path import join

from narps_open.utils.configuration import Configuration
from narps_open.data.results import ResultsCollectionFactory
from narps_open.pipelines import implemented_pipelines

dataset_dir = Configuration()['directories']['narps_results']
collection_factory = ResultsCollectionFactory()

for team_id, _ in implemented_pipelines.items():

    # Init collection
    collection = collection_factory.get_collection(team_id)

    # Create download directory if not existing
    # makedirs(collection.directory, exist_ok = True)
    print(f'mkdir -p {collection.directory}')

    # Create dataset entries
    base_directory = collection.directory.replace(dataset_dir, '')
    for file_name, file_url in collection.files.items():
        command = f'datalad download-url --nosave\
        --path {join(base_directory, file_name+".nii.gz")} --dataset {dataset_dir} {file_url}'
        print(command)
