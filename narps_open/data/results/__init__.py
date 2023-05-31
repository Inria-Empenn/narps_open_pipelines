#!/usr/bin/python
# coding: utf-8

""" This module allows to get Neurovault collections corresponding
    to results from teams involed in NARPS
"""

from os import remove, makedirs
from os.path import join
from json import loads
from zipfile import ZipFile
from urllib.request import urlretrieve
from argparse import ArgumentParser
from importlib_resources import files

from narps_open.utils.configuration import Configuration
from narps_open.data.description import TeamDescription
from narps_open.pipelines import implemented_pipelines
from narps_open.utils import show_download_progress

class ResultsCollection():
    """ Represents a Neurovault collections corresponding
        to results from teams involed in NARPS.
    """

    def __init__(self, team_id: str):

        # Initialize attributes
        self.team_id = team_id
        description = TeamDescription(team_id = self.team_id)
        self.id = description.general['NV_collection_link'].split('/')[-2]
        self.url = description.general['NV_collection_link'] + 'download'
        self.directory = join(
            Configuration()['directories']['narps_results'],
            'orig',
            self.id + '_' + self.team_id
            )
        self.files = {
            'hypo1_thresh.nii.gz' : 'hypo1_thresh.nii.gz',
            'hypo1_unthresh.nii.gz' : 'hypo1_unthresh.nii.gz',
            'hypo2_thresh.nii.gz' : 'hypo2_thresh.nii.gz',
            'hypo2_unthresh.nii.gz' : 'hypo2_unthresh.nii.gz',
            'hypo3_thresh.nii.gz' : 'hypo3_thresh.nii.gz',
            'hypo3_unthresh.nii.gz' : 'hypo3_unthresh.nii.gz',
            'hypo4_thresh.nii.gz' : 'hypo4_thresh.nii.gz',
            'hypo4_unthresh.nii.gz' : 'hypo4_unthresh.nii.gz',
            'hypo5_thresh.nii.gz' : 'hypo5_thresh.nii.gz',
            'hypo5_unthresh.nii.gz' : 'hypo5_unthresh.nii.gz',
            'hypo6_thresh.nii.gz' : 'hypo6_thresh.nii.gz',
            'hypo6_unthresh.nii.gz' : 'hypo6_unthresh.nii.gz',
            'hypo7_thresh.nii.gz' : 'hypo7_thresh.nii.gz',
            'hypo7_unthresh.nii.gz' : 'hypo7_unthresh.nii.gz',
            'hypo8_thresh.nii.gz' : 'hypo8_thresh.nii.gz',
            'hypo8_unthresh.nii.gz' : 'hypo8_unthresh.nii.gz',
            'hypo9_thresh.nii.gz' : 'hypo9_thresh.nii.gz',
            'hypo9_unthresh.nii.gz' : 'hypo9_unthresh.nii.gz'
        }

        # Make correspondences between the names given by the
        # team in neurovault collections, and the expected names of the hypotheses files.
        if Configuration()['results']['neurovault_naming']:
            with open(join(files('narps_open.data.results'),'results.json'), 'r') as file:
                neurovault_files = loads(file.read())[self.team_id]

            if neurovault_files:
                self.files = neurovault_files

    def download(self):
        """ Download the collection, unzip it and remove zip file. """

        # Create download directory if not existing
        makedirs(self.directory, exist_ok = True)

        # Download dataset
        print('Collecting results for team', self.team_id)
        zip_filename = join(self.directory, 'NARPS-'+self.team_id+'.zip')
        urlretrieve(self.url, zip_filename, show_download_progress)

        # Unzip files directly in the download directory
        with ZipFile(zip_filename, 'r') as zip_file:
            for zip_info in zip_file.infolist():
                zip_info.filename = zip_info.filename.split('/')[-1]
                zip_info.filename = join(self.directory, zip_info.filename)
                zip_file.extract(zip_info)

        # Remove zip file
        remove(zip_filename)

if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser(description='Get Neurovault collection of results from NARPS teams.')
    group = parser.add_mutually_exclusive_group(required = True)
    group.add_argument('-t', '--teams', nargs='+', type=str, action='extend',
        help='a list of team IDs')
    group.add_argument('-a', '--all', action='store_true', help='download results from all teams')
    arguments = parser.parse_args()

    if arguments.all:
        for team_id, _ in implemented_pipelines.items():
            ResultsCollection(team_id).download()
    else:
        for team in arguments.teams:
            ResultsCollection(team).download()
