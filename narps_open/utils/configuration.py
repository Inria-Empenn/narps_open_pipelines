#!/usr/bin/python
# coding: utf-8

""" Accessing information about teams from the analysis_pipelines_full_configuration.tsv file """

from os.path import join
from csv import DictReader
from argparse import ArgumentParser
from json import dumps
from importlib_resources import files

class TeamConfiguration(dict):
    """ This class allows to access information about a NARPS team
        Arguments:
        - team_id : str, the ID of the team to collect information from
    """

    configuration_file = join(
        files('narps_open.pipelines'),
        'analysis_pipelines_full_configuration.tsv')
    derived_configuration_file = join(
        files('narps_open.pipelines'),
        'analysis_pipelines_derived_configuration.tsv')

    def __init__(self, team_id):
        super().__init__()
        self.team_id = team_id
        self._load()

    @property
    def general(self) -> dict:
        """ Getter for the sub dictionary general """
        return self._get_sub_dict('general')

    @property
    def exclusions(self) -> dict:
        """ Getter for the sub dictionary exclusions """
        return self._get_sub_dict('exclusions')

    @property
    def preprocessing(self) -> dict:
        """ Getter for the sub dictionary preprocessing """
        return self._get_sub_dict('preprocessing')

    @property
    def analysis(self) -> dict:
        """ Getter for the sub dictionary analysis """
        return self._get_sub_dict('analysis')

    @property
    def categorized_for_analysis(self) -> dict:
        """ Getter for the sub dictionary categorized_for_analysis """
        return self._get_sub_dict('categorized_for_analysis')

    @property
    def derived(self) -> dict:
        """ Getter for the sub dictionary containing derived team configuration """
        return self._get_sub_dict('derived')

    def _get_sub_dict(self, key_first_part:str) -> dict:
        """ Return a sub-dictionary of self, with keys that contain key_first_part.
            The first part of the keys are removed, e.g.:
                'general.teamID' becomes 'teamID'
                'analysis.multiple_testing_correction' becomes 'multiple_testing_correction'
                'categorized_for_analysis.smoothing_coef' becomes 'smoothing_coef'
        """
        return {
            key.replace(key_first_part+'.',''):value
            for key, value in self.items() if key.startswith(key_first_part)
            }

    def _load(self):
        """ Load the contents of TeamConfiguration from the csv files.
            In this method, we parse the first two line of the csv configuration_file.
            These lines are the identifiers for each column of the file.
            NB: first line is the identifier of a group of columns.
            We transform the information in the two first lines as keys for the
            dictionary, so that the key is in the form :
                'first_line_identifier.second_line_identifier'
            This gives -for example- the following keys for the dictionary:
                'general.teamID'
                'analysis.multiple_testing_correction'
                'categorized_for_analysis.smoothing_coef'
                ...
            Furthermore, we parse the csv derived_configuration_file.
            The first line of this file being already a second level identifier,
            the firest level identifier will always be 'derived'.
            This gives -for example- the following key for the dictionary:
                'derived.n_participants'
        """

        # Parsing first file : self.configuration_file
        with open(self.configuration_file, newline='', encoding='utf-8') as csv_file:
            # Prepare first line (whose elements are first part of the keys)
            first_line = csv_file.readline().lower().replace('\n','').split('\t')
            for element_id, element in enumerate(first_line):
                if element == '':
                    first_line[element_id] = first_line[element_id - 1]

            # Prepare second line (whose elements are second part of the keys)
            second_line = csv_file.readline().replace('\n','').split('\t')

            # Read the whole file as a dict
            reader = DictReader(
                csv_file,
                fieldnames = [k1 + '.' + k2 for k1, k2 in zip(first_line, second_line)],
                delimiter = '\t'
                )

            # Update self with the key/value pairs from the corresponding line of the file
            found = False
            for row in reader:
                if row['general.teamID'] == self.team_id:
                    found = True
                    self.update(row)
                    break

            # If team id was not found in the file
            if not found:
                raise AttributeError(f'Team {self.team_id} was not found in the configuration.')

        # Parsing second file : self.derived_configuration_file
        with open(self.derived_configuration_file, newline='', encoding='utf-8') as csv_file:
            # Prepare first line (whose elements are second part of the keys)
            first_line = csv_file.readline().replace('\n','').split('\t')

            # Read the rest of the file as a dict
            reader = DictReader(
                csv_file,
                fieldnames = ['derived.' + k2 for k2 in first_line],
                delimiter = '\t'
                )

            # Update self with the key/value pairs from the file
            found = False
            for row in reader:
                if row['derived.teamID'] == self.team_id:
                    found = True
                    row.pop('derived.teamID', None) # Remove useless 'derived.teamID' key
                    self.update(row)
                    break

            # If team id was not found in the file
            if not found:
                raise AttributeError(f'Team {self.team_id}\
                    was not found in the derived configuration.')

if __name__ == '__main__':

    # Parse arguments
    parser = ArgumentParser(description='Get information from a NARPS team.')
    parser.add_argument('-t', '--team', type=str, required=True,
        help='the team ID')
    parser.add_argument('-d', '--dictionary', type=str, required=False,
        choices=[
            'general',
            'exclusions',
            'preprocessing',
            'analysis',
            'categorized_for_analysis',
            'derived'
            ],
        help='the sub dictionary of team information')
    arguments = parser.parse_args()

    # Initialize a TeamConfiguration
    information = TeamConfiguration(team_id = arguments.team)

    if arguments.dictionary == 'general':
        print(dumps(information.general, indent = 4))
    elif arguments.dictionary == 'exclusions':
        print(dumps(information.exclusions, indent = 4))
    elif arguments.dictionary == 'preprocessing':
        print(dumps(information.preprocessing, indent = 4))
    elif arguments.dictionary == 'analysis':
        print(dumps(information.analysis, indent = 4))
    elif arguments.dictionary == 'categorized_for_analysis':
        print(dumps(information.categorized_for_analysis, indent = 4))
    elif arguments.dictionary == 'derived':
        print(dumps(information.derived, indent = 4))
    else:
        print(dumps(information, indent = 4))
