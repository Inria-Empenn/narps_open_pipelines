#!/usr/bin/python
# coding: utf-8

""" This module allows to run pipelines from NARPS open. """

from os.path import isfile
from importlib import import_module
from random import choices
from argparse import ArgumentParser

from nipype import Workflow, config

from narps_open.pipelines import Pipeline, implemented_pipelines
from narps_open.data.participants import (
    get_all_participants,
    get_participants,
    get_participants_subset
    )
from narps_open.utils.configuration import Configuration
from narps_open.pipelines import get_implemented_pipelines

class PipelineRunner():
    """ A class that allows to run a NARPS pipeline. """

    def __init__(self, team_id: str = '') -> None:
        self._pipeline = None

        # Set team_id. It's important to use the property setter here,
        # so that the code inside it is executed. That would not be the
        # case if simply setting the `self._team_id` attribute i.e.: `self._team_id = team_id`
        self.team_id = team_id

    @property
    def pipeline(self) -> Pipeline:
        """ Getter for property pipeline """
        return self._pipeline

    @property
    def subjects(self) -> list:
        """ Getter for property subjects """
        return self._pipeline.subject_list

    @subjects.setter
    def subjects(self, value: list) -> None:
        """ Setter for property subjects """

        all_participants = get_all_participants()
        for subject_id in value:
            if str(int(subject_id)).zfill(3) not in all_participants:
                raise AttributeError(f'Subject ID {subject_id} is not valid')

        self._pipeline.subject_list = list(dict.fromkeys(
            [str(int(subject_id)).zfill(3) for subject_id in value])) # remove duplicates

    @subjects.setter
    def random_nb_subjects(self, value: int) -> None:
        """ Setter for property random_nb_subjects """
        # Generate a random list of subjects
        self._pipeline.subject_list = choices(get_participants(self.team_id), k = value)

    @subjects.setter
    def nb_subjects(self, value: int) -> None:
        """ Setter for property nb_subjects """
        # Get a subset of participants
        self._pipeline.subject_list = get_participants_subset(value)

    @property
    def team_id(self) -> str:
        """ Getter for property team_id """
        return self._team_id

    @team_id.setter
    def team_id(self, value: str) -> None:
        """ Setter for property random_nb_subjects """
        self._team_id = value

        # It's up to the PipelineRunner to find the right pipeline, based on the team ID
        if self._team_id not in implemented_pipelines:
            raise KeyError(f'Wrong team ID : {self.team_id}')

        if implemented_pipelines[self._team_id] is None:
            raise NotImplementedError(f'Pipeline not implemented for team : {self.team_id}')

        # Instantiate the pipeline
        class_type = getattr(
            import_module('narps_open.pipelines.team_'+self._team_id),
            implemented_pipelines[self._team_id])
        self._pipeline = class_type()

    def start(self, first_level_only: bool = False, group_level_only: bool = False) -> None:
        """
        Start the pipeline

        Arguments:
            - first_level_only: bool (False by default), run the first level workflows only,
                (= preprocessing + run level + subject_level)
            - group_level_only: bool (False by default), run the group level workflows only
        """
        # Set global nipype config for pipeline execution
        config.update_config(dict(execution = {'stop_on_first_crash': 'True'}))

        # Disclaimer
        print('Starting pipeline for team: '+
            f'{self.team_id}, with {len(self.subjects)} subjects: {self.subjects}')

        if first_level_only and group_level_only:
            raise AttributeError('first_level_only and group_level_only cannot both be True')

        # Generate workflow lists
        first_level_workflows = []
        group_level_workflows = []

        if not group_level_only:
            for workflow in [
                self._pipeline.get_preprocessing(),
                self._pipeline.get_run_level_analysis(),
                self._pipeline.get_subject_level_analysis()]:

                if isinstance(workflow, list):
                    for sub_workflow in workflow:
                        first_level_workflows.append(sub_workflow)
                else:
                    first_level_workflows.append(workflow)

        if not first_level_only:
            for workflow in [self._pipeline.get_group_level_analysis()]:
                if isinstance(workflow, list):
                    for sub_workflow in workflow:
                        group_level_workflows.append(sub_workflow)
                else:
                    group_level_workflows.append(workflow)

        # Launch workflows
        for workflow in first_level_workflows + group_level_workflows:
            if workflow is None:
                pass
            elif not isinstance(workflow, Workflow):
                raise AttributeError('Workflow must be of type nipype.Workflow')
            else:
                nb_procs = Configuration()['runner']['nb_procs']
                if nb_procs > 1:
                    workflow.run('MultiProc', plugin_args = {'n_procs': nb_procs})
                else:
                    workflow.run()

    def get_missing_first_level_outputs(self):
        """ Return the list of missing files after computations of the first level """
        files = self._pipeline.get_preprocessing_outputs()
        files += self._pipeline.get_run_level_outputs()
        files += self._pipeline.get_subject_level_outputs()

        return [f for f in files if not isfile(f)]

    def get_missing_group_level_outputs(self):
        """ Return the list of missing files after computations of the group level """
        files = self._pipeline.get_group_level_outputs()

        return [f for f in files if not isfile(f)]

def main():
    """ Entry-point for the command line tool narps_open_runner """

    # Parse arguments
    parser = ArgumentParser(description='Run the pipelines from NARPS.')
    parser.add_argument('-t', '--team', type=str, required=True,
        help='the team ID', choices=get_implemented_pipelines())
    subjects = parser.add_mutually_exclusive_group(required=True)
    subjects.add_argument('-s', '--subjects', nargs='+', type=str, action='extend',
        help='a list of subjects to be selected')
    subjects.add_argument('-n', '--nsubjects', type=str,
        help='the number of subjects to be selected')
    subjects.add_argument('-r', '--rsubjects', type=str,
        help='the number of subjects to be selected randomly')
    levels = parser.add_mutually_exclusive_group(required=False)
    levels.add_argument('-g', '--group', action='store_true', default=False,
        help='run the group level only')
    levels.add_argument('-f', '--first', action='store_true', default=False,
        help='run the first levels only (preprocessing + subjects + runs)')
    parser.add_argument('-c', '--check', action='store_true', required=False,
        help='check pipeline outputs (runner is not launched)')
    arguments = parser.parse_args()

    # Initialize a PipelineRunner
    runner = PipelineRunner(team_id = arguments.team)
    runner.pipeline.directories.dataset_dir = Configuration()['directories']['dataset']
    runner.pipeline.directories.results_dir = Configuration()['directories']['reproduced_results']
    runner.pipeline.directories.set_output_dir_with_team_id(arguments.team)
    runner.pipeline.directories.set_working_dir_with_team_id(arguments.team)

    # Handle subject
    if arguments.subjects is not None:
        runner.subjects = arguments.subjects
    elif arguments.rsubjects is not None:
        runner.random_nb_subjects = int(arguments.rsubjects)
    else:
        runner.nb_subjects = int(arguments.nsubjects)

    # Check data
    if arguments.check:
        print('Missing files for team', arguments.team, 'after running',
            len(runner.pipeline.subject_list), 'subjects:')
        if not arguments.group:
            print('First level:', runner.get_missing_first_level_outputs())
        if not arguments.first:
            print('Group level:', runner.get_missing_group_level_outputs())

    # Start the runner
    else:
        runner.start(arguments.first, arguments.group)

if __name__ == '__main__':
    main()
