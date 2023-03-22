#!/usr/bin/python
# coding: utf-8

""" This module allows to get Neurovault corresponding to results from teams involed in NARPS """

from os import remove, makedirs
from os.path import join
from zipfile import ZipFile
from urllib.request import urlretrieve
from argparse import ArgumentParser

from narps_open.utils.configuration import Configuration
from narps_open.utils.description import TeamDescription
from narps_open.pipelines import implemented_pipelines

def show_progress(count, block_size, total_size):
    """ A hook function to be passed to urllib.request.urlretrieve in order to
        print the progress of a download.

        Arguments:
        - count: int - the number of blocks already downloaded
        - block_size: int - the size in bytes of a block
        - total_size: int - the total size in bytes of the download. -1 if not provided.
    """
    if total_size != -1:
        # Display a percentage
        display_value = str(int(count * block_size * 100 / total_size))+' %'
    else:
        # Draw a pretty cursor
        cursor = ['⣾','⣽','⣻','⢿','⡿','⣟','⣯','⣷']
        display_value = cursor[int(count)%len(cursor)]

    # Showing download progress
    print('Downloading', display_value, end='\r')

def download_result_collection(team_id: str):
    """ Download a Neurovault collection corresponding to results from a team involed in NARPS.
        Unzip it and remove zip file.

        Arguments:
        - team_id: team corresponding to the requested collection
    """
    # Get collection url
    description = TeamDescription(team_id = team_id)
    url = description.general['NV_collection_link'] + '/download'

    # Create download directory if not existing
    download_directory = Configuration()['directories']['narps_results']
    makedirs(download_directory, exist_ok = True)

    # Download dataset
    print('Collecting results for team', team_id)
    zip_filename = join(download_directory, 'NARPS-'+team_id+'.zip')
    urlretrieve(url, zip_filename, show_progress)

    # Unzip file
    zip_extact_folder = join(download_directory)
    with ZipFile(zip_filename, 'r') as zip_file:
        zip_file.extractall(zip_extact_folder)

    # Remove zip file
    remove(zip_filename)

def download_all_result_collections():
    """ Download all Neurovault collections corresponding to results from teams involed in NARPS.
    """
    for team_id, _ in implemented_pipelines.items():
        download_result_collection(team_id)

if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser(description='Get Neurovault collection of results from NARPS teams.')
    group = parser.add_mutually_exclusive_group(required = True)
    group.add_argument('-t', '--teams', nargs='+', type=str, action='extend',
        help='a list of team IDs')
    group.add_argument('-a', '--all', action='store_true', help='download results from all teams')
    arguments = parser.parse_args()

    if arguments.all is True:
        download_all_result_collections()
    else:
        for team in arguments.teams:
            download_result_collection(team)
