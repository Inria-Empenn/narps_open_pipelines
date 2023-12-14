#!/usr/bin/python
# coding: utf-8

""" A set of functions to get the participants data for the narps_open package """

from os.path import join

from pandas import read_csv

from narps_open.data.description import TeamDescription
from narps_open.utils.configuration import Configuration

def get_participants_information():
    """ Get a list of participants information from the tsv file from NARPS """
    return read_csv(join(Configuration()['directories']['dataset'], 'participants.tsv'), sep='\t')

def get_all_participants() -> list:
    """ Return a list of all participants included in NARPS.
        This list is ordered so that subsets of 20, 40, 60, 80, 108 participants
        are balanced in terms of belonging to the equal indifference and equal
        range groups.
    """
    return [
        '020', '001', '070', '013', '120', '109', '118', '035', '002', '025',
        '018', '053', '046', '073', '066', '121', '098', '011', '116', '087',
        '008', '069', '106', '095', '004', '113', '104', '115', '092', '089',
        '090', '045', '016', '117', '124', '093', '088', '021', '094', '041',
        '062', '017', '040', '083', '084', '107', '056', '119', '064', '103',
        '044', '057', '060', '061', '112', '085', '050', '027', '082', '059',
        '022', '019', '052', '047', '030', '039', '100', '029', '108', '067',
        '096', '009', '058', '055', '024', '015', '080', '077', '006', '003',
        '076', '072', '014', '102', '010', '074', '038', '114', '026', '079',
        '054', '071', '032', '051', '110', '081', '036', '037', '068', '099',
        '105', '063', '075', '033', '049', '123', '043', '005']

def get_participants(team_id: str) -> list:
    """ Return a list of participants that were taken into account by a given team

    Args:
        team_id: str, the ID of the team.

    Returns: a list of participants labels
    """
    description = TeamDescription(team_id)
    excluded_participants = description.derived['excluded_participants'].replace(' ','').split(',')

    return [p for p in get_all_participants() if p not in excluded_participants]

def get_participants_subset(nb_participants: int = 108) -> list:
    """ Return a list of participants of length nb_participants """
    return get_all_participants()[0:nb_participants]

def get_group(group_name: str) -> list:
    """ Return a list containing all the participants inside the group_name group """

    participants = get_participants_information()
    group = participants.loc[participants['group'] == group_name]['participant_id'].values.tolist()

    return [p.replace('sub-', '') for p in group]
