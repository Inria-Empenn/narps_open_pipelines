#!/usr/bin/python
# coding: utf-8

""" This module allows to run pipelines from NARPS open. """

from importlib import import_module
from random import choices
from argparse import ArgumentParser
from pathlib import Path

from nipype import Workflow

from narps_open.pipelines import Pipeline, implemented_pipelines

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

        for subject in value:
            if int(subject) > 108:
                raise AttributeError(f'Subject ID {subject} is not in the range [1:108]')

        self._pipeline.subject_list = list(dict.fromkeys(
            [str(int(subject_id)).zfill(3) for subject_id in value])) # remove duplicates

    @subjects.setter
    def random_nb_subjects(self, value: int) -> None:
        """ Setter for property random_nb_subjects """
        # Generate a random list of subjects
        self._pipeline.subject_list = choices(
            [str(subject_id).zfill(3) for subject_id in range(1,109)], # a list of all subjects
            k = value)

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

        # Instanciate the pipeline
        class_type = getattr(
            import_module('narps_open.pipelines.team_'+self._team_id),
            implemented_pipelines[self._team_id])
        self._pipeline = class_type()

    def start(self) -> None:
        """
        Start the pipeline
        """
        print('Starting pipeline for team: '+
            f'{self.team_id}, with {len(self.subjects)} subjects: {self.subjects}')

        for workflow in [
            self._pipeline.get_preprocessing(),
            self._pipeline.get_run_level_analysis(),
            self._pipeline.get_subject_level_analysis(),
            self._pipeline.get_group_level_analysis()
        ]:
            if workflow is None:
                pass
            elif isinstance(workflow, list):
                for sub_workflow in workflow:
                    if not isinstance(sub_workflow, Workflow):
                        raise AttributeError('Workflow must be of type nipype.Workflow')
                    sub_workflow.run('MultiProc', plugin_args={'n_procs': 8})
                    #sub_workflow.run()
            else:
                if not isinstance(workflow, Workflow):
                    raise AttributeError('Workflow must be of type nipype.Workflow')
                workflow.run('MultiProc', plugin_args={'n_procs': 8})
                #workflow.run()

if __name__ == '__main__':

    # Parse arguments
    parser = ArgumentParser(description='Run the pipelines from NARPS.')
    parser.add_argument('-t', '--team', type=str, required=True,
        help='the team ID')
    parser.add_argument('-d', '--dataset', type=Path, required=True,
        help='the path to the ds001734 dataset')
    parser.add_argument('-o', '--output', type=Path, required=True,
        help='the path to store the output files')
    subjects = parser.add_mutually_exclusive_group(required=True)
    subjects.add_argument('-r', '--random', type=str,
        help='the number of subjects to be randomly selected')
    subjects.add_argument('-s', '--subjects', nargs='+', type=str, action='extend',
        help='a list of subjects')
    arguments = parser.parse_args()

    # Initialize a PipelineRunner
    runner = PipelineRunner(team_id = arguments.team)
    runner.pipeline.directories.dataset_dir = arguments.dataset
    runner.pipeline.directories.results_dir = arguments.output
    runner.pipeline.directories.set_output_dir_with_team_id(arguments.team)
    runner.pipeline.directories.set_working_dir_with_team_id(arguments.team)

    if arguments.subjects is not None:
        runner.subjects = arguments.subjects
    else:
        runner.random_nb_subjects = int(arguments.random)

    # Start the runner
    runner.start()
