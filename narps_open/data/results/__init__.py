#!/usr/bin/python
# coding: utf-8

""" This module allows to get Neurovault collections corresponding
    to results from teams involved in NARPS
"""

from os import makedirs
from os.path import join
from importlib import import_module
from json import loads
from urllib.request import urlretrieve, urlopen

from narps_open.utils.configuration import Configuration
from narps_open.data.description import TeamDescription
from narps_open.utils import show_download_progress

class ResultsCollectionFactory():
    """ A factory class to instantiate ResultsCollection objects """
    collections = {
        '2T6S': 'ResultsCollection2T6S'
    }

    def get_collection(self, team_id):
        """ Return a ResultsCollection object or specialized child class if available """
        # Send the default ResultsCollection class
        if team_id not in ResultsCollectionFactory.collections:
            return ResultsCollection(team_id)

        # There is a specialized class for this team id
        collection_class = getattr(
            import_module(f'narps_open.data.results.team_{team_id}'),
            ResultsCollectionFactory.collections[team_id]
            )
        return collection_class()

class ResultsCollection():
    """ Represents a Neurovault collections corresponding
        to results from teams involved in NARPS.
    """

    def __init__(self, team_id: str):
        # Initialize attributes
        self.team_id = team_id
        self.uid = self.get_uid()
        self.directory = join(
            Configuration()['directories']['narps_results'],
            'orig',
            self.uid + '_' + self.team_id
            )
        self.files = self.get_file_urls()

    def get_uid(self):
        """ Return the uid of the collection by browsing the team description """
        return TeamDescription(team_id = self.team_id).general['NV_collection_link'].split('/')[-2]

    def get_file_urls(self):
        """ Return a dict containing the download url for each file of the collection.
        * dict key is the file base name (with extension)
        * dict value is the download url for the file on Neurovault
        """

        # Get the images data from Neurovault's API
        collection_url = 'https://neurovault.org/api/collections/' + self.uid + '/images/'

        with urlopen(collection_url) as response:
            json = loads(response.read())

            file_urls = {}
            for result in json['results']:
                # Get data for a file in the collection
                file_urls[result['name']+'.nii.gz'] = result['file']

        return file_urls

    def download(self):
        """ Download the collection, file by file. """

        # Create download directory if not existing
        makedirs(self.directory, exist_ok = True)

        # Download dataset
        print('Collecting results for team', self.team_id)
        for file_name, file_url in self.files.items():
            urlretrieve(
                file_url,
                join(self.directory, file_name),
                show_download_progress
                )

    def rectify(self):
        """ Rectify files in the collection, if needed.
        This method can be overwritten by child classes.
        """
        # Nothing to rectify by default
