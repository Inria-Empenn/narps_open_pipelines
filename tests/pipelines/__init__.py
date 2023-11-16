#!/usr/bin/python
# coding: utf-8

"""
Configuration for testing of the narps_open.pipelines modules.
"""

from pytest import helpers

@helpers.register
def mock_event_data(mocker):
    """ Mocks the return of the open function with the contents of a fake event file """

    fake_event_data = 'onset duration\tgain\tloss\tRT\tparticipant_response\n'
    fake_event_data += '4.071\t4\t14\t6\t2.388\tweakly_accept\n'
    fake_event_data += '11.834\t4\t34\t14\t2.289\tstrongly_accept\n'
    fake_event_data += '19.535\t4\t38\t19\t0\tNoResp\n'
    fake_event_data += '27.535\t4\t10\t15\t2.08\tstrongly_reject\n'
    fake_event_data += '36.435\t4\t16\t17\t2.288\tweakly_reject\n'

    mocker.patch('builtins.open', mocker.mock_open(read_data = fake_event_data))

@helpers.register
def mock_participants_data(mocker):
    """ Mocks the return of the open function with the contents of a fake participants file """

    fake_participants_data = 'participant_id\tgroup\tgender\tage\n'
    fake_participants_data += 'sub-001\tequalIndifference\tM\t24\n'
    fake_participants_data += 'sub-002\tequalRange\tM\t25\n'
    fake_participants_data += 'sub-003\tequalIndifference\tF\t27\n'
    fake_participants_data += 'sub-004\tequalRange\tM\t25\n'

    mocker.patch('builtins.open', mocker.mock_open(read_data = fake_participants_data))
