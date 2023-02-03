#!/usr/bin/python
# coding: utf-8

""" Tests of the 'runner' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_runner.py
    pytest -q test_runner.py -k <selected_test>
"""

from pytest import raises

from narps_open.runner import PipelineRunner
from narps_open.pipelines.team_2T6S import PipelineTeam2T6S

class TestPipelineRunner:
    """ A class that contains all the unit tests for the PipelineRunner class."""

    @staticmethod
    def test_create():
        """ Test the creation of a PipelineRunner object """

        # 1 - Instanciate a runner without team id
        with raises(KeyError):
            PipelineRunner()

        # 2 - Instanciate a runner with wrong team id
        with raises(KeyError):
            PipelineRunner('wrong_id')

        # 3 - Instanciate a runner with a not implemented team id
        with raises(NotImplementedError):
            PipelineRunner('08MQ')

        # 4 - Instanciate a runner with an implemented team id
        runner = PipelineRunner('2T6S')
        assert isinstance(runner._pipeline, PipelineTeam2T6S)

    @staticmethod
    def test_subjects():
        """ Test the PipelineRunner features of building subject lists """
        runner = PipelineRunner('2T6S')

        # 1 - random list
        # Check the right number of subjects is generated
        runner.random_nb_subjects = 8
        assert len(runner.subjects) == 8
        runner.random_nb_subjects = 4
        assert len(runner.subjects) == 4

        # Check formatting and consistency
        for subject in runner.subjects:
            assert isinstance(subject, str)
            assert len(subject) == 3
            assert int(subject) > 0
            assert int(subject) < 109

        # 2 - fixed list
        # Check subject ids too high
        with raises(AttributeError):
            runner.subjects = ['120', '042']

        # Check duplicate subject ids are removed
        runner.subjects = ['043', '042', '042', '045']
        assert runner.subjects == ['043', '042', '045']

        # Check formatting
        runner.subjects = [42, '00042', '0043', 45]
        assert runner.subjects == ['042', '043', '045']

    @staticmethod
    def test_directories():
        """ Test PipelineRunner handling directories """

    @staticmethod
    def test_start():
        """ Test the start method of PipelineRunner """
