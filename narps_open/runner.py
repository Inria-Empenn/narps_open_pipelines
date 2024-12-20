#!/usr/bin/python
# coding: utf-8

""" This module allows to run pipelines from NARPS open. """

from os.path import isfile
from importlib import import_module
from random import choices
from argparse import ArgumentParser
from enum import Flag, auto

from nipype import Workflow, config

from narps_open.pipelines import Pipeline, implemented_pipelines
from narps_open.data.participants import (
    get_all_participants,
    get_participants,
    get_participants_subset
    )
from narps_open.utils.configuration import Configuration
from narps_open.pipelines import get_implemented_pipelines

class PipelineRunnerLevel(Flag):
    """ A class to enumerate possible levels for a pipeline. """
    NONE = 0
    PREPROCESSING = auto()
    RUN = auto()
    SUBJECT = auto()
    GROUP = auto()
    ALL = PREPROCESSING | RUN | SUBJECT | GROUP
    FIRST = PREPROCESSING | RUN | SUBJECT
    SECOND = GROUP

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

    @staticmethod
    def get_workflows(input_workflow):
        """
        Return a list of nipype.Workflow from the passed argument.

        Arguments:
            - input_workflow: either a list of nipype.Workflow or a nipype.Workflow

        Returns:
            - a list of nipype.Worflow:
                - [input_workflow] if input_workflow is a nipype.Workflow
                - input_workflow if input_workflow is a list of nipype.Workflow
            - an empty list if input_workflow is None
        """
        if isinstance(input_workflow, Workflow):
            return [input_workflow]
        if input_workflow is None:
            return []
        if isinstance(input_workflow, list):
            for sub_workflow in input_workflow:
                if not isinstance(sub_workflow, Workflow):
                    raise AttributeError('When using a list of workflows,\
all elements must be of type nipype.Workflow')
            return input_workflow
        raise AttributeError('Workflow must be of type list or nipype.Workflow')

    def start(self, level: PipelineRunnerLevel = PipelineRunnerLevel.ALL) -> None:
        """
        Start the pipeline

        Arguments:
            - level: PipelineRunnerLevel, indicates which workflow(s) to run
        """
        # Set global nipype config for pipeline execution
        config.update_config(dict(execution = {'stop_on_first_crash': 'True'}))

        # Disclaimer
        print('Starting pipeline for team: '+
            f'{self.team_id}, with {len(self.subjects)} subjects: {self.subjects}')

        # Generate workflow list
        workflows = []
        if bool(level & PipelineRunnerLevel.PREPROCESSING):
            workflows += self.get_workflows(self._pipeline.get_preprocessing())
        if bool(level & PipelineRunnerLevel.RUN):
            workflows += self.get_workflows(self._pipeline.get_run_level_analysis())
        if bool(level & PipelineRunnerLevel.SUBJECT):
            workflows += self.get_workflows(self._pipeline.get_subject_level_analysis())
        if bool(level & PipelineRunnerLevel.GROUP):
            workflows += self.get_workflows(self._pipeline.get_group_level_analysis())

        # Launch workflows
        for workflow in workflows:
            nb_procs = Configuration()['runner']['nb_procs']
            if nb_procs > 1:
                workflow.run('MultiProc', plugin_args = {'n_procs': nb_procs})
            else:
                workflow.run()

    def get_missing_outputs(self, level: PipelineRunnerLevel = PipelineRunnerLevel.ALL):
        """
        Return the list of missing files after computations of the level(s)
        
        Arguments:
            - level: PipelineRunnerLevel, indicates for which workflow(s) to search output files
        """
        # Disclaimer
        print('Missing files for team: '+
            f'{self.team_id}, with {len(self.subjects)} subjects: {self.subjects}')

        # Generate files list
        files = []
        if bool(level & PipelineRunnerLevel.PREPROCESSING):
            files += self._pipeline.get_preprocessing_outputs()
        if bool(level & PipelineRunnerLevel.RUN):
            files += self._pipeline.get_run_level_outputs()
        if bool(level & PipelineRunnerLevel.SUBJECT):
            files += self._pipeline.get_subject_level_outputs()
        if bool(level & PipelineRunnerLevel.GROUP):
            files += self._pipeline.get_group_level_outputs()

        # Return non existing files
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
    levels = parser.add_argument_group('Pipeline runner levels')
    levels.add_argument('-al', '--all_levels', action='store_true', default=False,
        help='run all levels analyses')
    levels.add_argument('-gl', '--group_level', action='store_true', default=False,
        help='run the group level analysis')
    levels.add_argument('-sl', '--subject_level', action='store_true', default=False,
        help='run the subject level analysis')
    levels.add_argument('-rl', '--run_level', action='store_true', default=False,
        help='run the run level analysis')
    levels.add_argument('-pl', '--preprocessing', action='store_true', default=False,
        help='run the preprocessing')
    parser.add_argument('-c', '--check', action='store_true', required=False,
        help='check pipeline outputs (runner is not launched)')
    parser.add_argument('-e', '--exclusions', action='store_true', required=False,
        help='run the analyses without the excluded subjects')
    arguments = parser.parse_args()

    # Check arguments
    if arguments.exclusions and not arguments.nsubjects:
        print('Argument -e/--exclusions only works with -n/--nsubjects')
        return

    # Initialize a PipelineRunner
    runner = PipelineRunner(team_id = arguments.team)
    runner.pipeline.directories.dataset_dir = Configuration()['directories']['dataset']
    runner.pipeline.directories.results_dir = Configuration()['directories']['reproduced_results']
    runner.pipeline.directories.set_output_dir_with_team_id(arguments.team)
    runner.pipeline.directories.set_working_dir_with_team_id(arguments.team)

    # Handle subjects
    if arguments.subjects is not None:
        runner.subjects = arguments.subjects
    elif arguments.rsubjects is not None:
        runner.random_nb_subjects = int(arguments.rsubjects)
    else:
        if arguments.exclusions:
            # Intersection between the requested subset and the list of not excluded subjects
            runner.subjects = list(
                set(get_participants_subset(int(arguments.nsubjects)))
              & set(get_participants(arguments.team))
            )
        else:
            runner.nb_subjects = int(arguments.nsubjects)

    # Build pipeline runner level
    if arguments.all_levels:
        level = PipelineRunnerLevel.ALL
    if arguments.preprocessing_level:
        level &= PipelineRunnerLevel.PREPROCESSING
    if arguments.run_level:
        level &= PipelineRunnerLevel.RUN
    if arguments.subject_level:
        level &= PipelineRunnerLevel.SUBJECT
    if arguments.group_level:
        level &= PipelineRunnerLevel.GROUP

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

        runner.start(level)

if __name__ == '__main__':
    main()
