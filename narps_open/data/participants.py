#!/usr/bin/python
# coding: utf-8

""" A set of functions to get the participants data for the narps_open package """

from pandas import read_csv

from narps_open.data.description import TeamDescription

def participants_tsv():
    """ Get a list of participants from the tsv file from NARPS """
    participants = join(directories()["exp"], "participants.tsv")

    return read_csv(participants, sep="\t")

def get_all_participants() -> list:
    """ Return a list of all participants included in NARPS.
        This list is ordered so that subsets of 20, 40, 60, 80, 108 participants
        are balanced in terms of belonging to the equal indifference and equal
        range groups.
    """
    return [
        '020', '070', '120', '118', '002', '018', '046', '066', '098', '116', '001', '013', '109',
        '035', '025', '053', '073', '121', '011', '087',
        '008', '106', '004', '104', '092', '090', '016', '124', '088', '094', '069', '095', '113',
        '115', '089', '045', '117', '093', '021', '041',
        '062', '040', '084', '056', '064', '044', '060', '112', '050', '082', '017', '083', '107',
        '119', '103', '057', '061', '085', '027', '059',
        '022', '052', '030', '100', '108', '096', '058', '024', '080', '006', '019', '047', '039',
        '029', '067', '009', '055', '015', '077', '003',
        '076', '014', '010', '038', '026', '054', '032', '110', '036', '068', '072', '102', '074',
        '114', '079', '071', '051', '081', '037', '099',
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
