#!/usr/bin/python
# coding: utf-8

""" Common functions to write pipelines """

def remove_file(_, file_name: str) -> None:
    """
    Fully remove files generated by a Node, once they aren't needed anymore.
    This function is meant to be used in a Nipype Function Node.

    Parameters:
    - _: input only used for triggering the Node
    - file_name: str, a single absolute filename of the file to remove
    """
    # This import must stay inside the function, as required by Nipype
    from os import remove

    try:
        remove(file_name)
    except OSError as error:
        print(error)

def remove_directory(_, directory_name: str) -> None:
    """
    Fully remove directory generated by a Node, once it is not needed anymore.
    This function is meant to be used in a Nipype Function Node.

    Parameters:
    - _: input only used for triggering the Node
    - directory_name: str, a single absolute path of the directory to remove
    """
    # This import must stay inside the function, as required by Nipype
    from shutil import rmtree

    rmtree(directory_name, ignore_errors = True)

def remove_parent_directory(_, file_name: str) -> None:
    """
    Fully remove directory generated by a Node, once it is not needed anymore.
    This function is meant to be used in a Nipype Function Node.

    Parameters:
    - _: input only used for triggering the Node
    - file_name: str, a single absolute path of a file : its parent directory is to remove
    """
    # This import must stay inside the function, as required by Nipype
    from pathlib import Path
    from shutil import rmtree

    rmtree(Path(file_name).parent.absolute(), ignore_errors = True)

def elements_in_string(input_str: str, elements: list) -> str: #| None:
    """
    Return input_str if it contains one element of the elements list.
    Return None otherwise.
    This function is meant to be used in a Nipype Function Node.

    Parameters:
    - input_str: str
    - elements: list of str, elements to be searched in input_str
    """
    if any(e in input_str for e in elements):
        return input_str
    return None

def clean_list(input_list: list, element = None) -> list:
    """
    Remove elements of input_list that are equal to element and return the resultant list.
    This function is meant to be used in a Nipype Function Node. It can be used inside a
    nipype.Workflow.connect call as well.

    Parameters:
    - input_list: list
    - element: any

    Returns:
    - input_list with elements equal to element removed
    """
    return [f for f in input_list if f != element]

def list_intersection(list_1: list, list_2: list) -> list:
    """
    Returns the intersection of two lists.
    This function is meant to be used in a Nipype Function Node. It can be used inside a
    nipype.Workflow.connect call as well.

    Parameters:
    - list_1: list
    - list_2: list

    Returns:
    - list, the intersection of list_1 and list_2
    """
    return [e for e in list_1 if e in list_2]

def list_to_file(input_list: list, file_name: str = 'elements.tsv') -> str:
    """
    Create a tsv file containing elements of the input list.
    This function is meant to be used in a Nipype Function Node.

    Parameters :
    - input_list: list

    Returns:
    - output_file: path to the created file
    """
    from os.path import abspath
    output_file = abspath(file_name)

    # Write un element per line
    with open(output_file, 'w') as writer:
        for element in input_list:
            writer.write(f'{element}\n')

    return output_file
