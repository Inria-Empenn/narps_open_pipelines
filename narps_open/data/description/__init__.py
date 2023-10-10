#!/usr/bin/python
# coding: utf-8

""" Accessing textual descriptions of the pipelines """

from os.path import join
from csv import DictReader
from json import dumps
from importlib_resources import files

class TeamDescription(dict):
    """ This class allows to access information about a NARPS team
        Arguments:
        - team_id : str, the ID of the team to collect information from
    """

    description_file = join(
        files('narps_open.data.description'),
        'analysis_pipelines_full_descriptions.tsv')
    derived_description_file = join(
        files('narps_open.data.description'),
        'analysis_pipelines_derived_descriptions.tsv')
    comments_description_file = join(
        files('narps_open.data.description'),
        'analysis_pipelines_comments.tsv')

    def __init__(self, team_id):
        super().__init__()
        self.team_id = team_id
        self._load()

    def __str__(self):
        return dumps(self, indent = 4)

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
        """ Getter for the sub dictionary containing derived team description """
        return self._get_sub_dict('derived')

    @property
    def comments(self) -> dict:
        """ Getter for the sub dictionary containing comments for NARPS Open Pipeline """
        return self._get_sub_dict('comments')

    def markdown(self):
        """ Return the team description as a string formatted in markdown """
        return_string = f'# NARPS team description : {self.team_id}\n'

        dictionaries = [
            self.general,
            self.exclusions,
            self.preprocessing,
            self.analysis,
            self.categorized_for_analysis,
            self.derived,
            self.comments
            ]

        names = [
            'General',
            'Exclusions',
            'Preprocessing',
            'Analysis',
            'Categorized for analysis',
            'Derived',
            'Comments'
            ]

        for dictionary, name in zip(dictionaries, names):
            return_string += f'## {name}\n'
            for key in dictionary:
                return_string += f'* `{key}` : {dictionary[key]}\n'

        return return_string

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
        """ Load the contents of TeamDescription from the csv files.
            In this method, we parse the first two line of the csv description_file.
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
            Furthermore, we parse the csv derived_description_file.
            The first line of this file being already a second level identifier,
            the first level identifier will always be 'derived'.
            This gives -for example- the following key for the dictionary:
                'derived.n_participants'
        """

        # Parsing first file : self.description_file
        with open(self.description_file, newline='', encoding='utf-8') as csv_file:
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
                raise AttributeError(f'Team {self.team_id} was not found in the description.')

        # Parsing second file : self.derived_description_file
        with open(self.derived_description_file, newline='', encoding='utf-8') as csv_file:
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
                    was not found in the derived description.')

        # Parsing third file : self.comments_description_file
        with open(self.comments_description_file, newline='', encoding='utf-8') as csv_file:
            # Prepare first line (whose elements are second part of the keys)
            first_line = csv_file.readline().replace('\n','').split('\t')

            # Read the rest of the file as a dict
            reader = DictReader(
                csv_file,
                fieldnames = ['comments.' + k2 for k2 in first_line],
                delimiter = '\t'
                )

            # Update self with the key/value pairs from the file
            found = False
            for row in reader:
                if row['comments.teamID'] == self.team_id:
                    found = True
                    row.pop('comments.teamID', None) # Remove useless 'comments.teamID' key
                    self.update(row)
                    break

            # If team id was not found in the file
            if not found:
                raise AttributeError(f'Team {self.team_id}\
                    was not found in the comments description.')
