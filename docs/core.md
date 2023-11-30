# Core functions you can use to write pipelines

Here are a few functions that could be useful for creating a pipeline with Nipype. These functions are meant to stay as unitary as possible.

These are intended to be inserted in a nipype.Workflow inside a [nipype.Function](https://nipype.readthedocs.io/en/latest/api/generated/nipype.interfaces.utility.wrappers.html#function) interface, or for some of them (see associated docstring) as part of a [nipype.Workflow.connect](https://nipype.readthedocs.io/en/latest/api/generated/nipype.pipeline.engine.workflows.html#nipype.pipeline.engine.workflows.Workflow.connect) method.

In the following example, we use the `list_intersection` function of `narps_open.core.common`, in both of the mentioned cases.

```python
from nipype import Node, Function, Workflow
from narps_open.core.common import list_intersection

# First case : a Function Node
intersection_node = Node(Function(
    function = list_intersection,
    input_names = ['list_1', 'list_2'],
    output_names = ['output']
    ), name = 'intersection_node')
intersection_node.inputs.list_1 = ['001', '002', '003', '004']
intersection_node.inputs.list_2 = ['002', '004', '005']
print(intersection_node.run().outputs.output) # ['002', '004']

# Second case : inside a connect node
# We assume that there is a node_0 returning ['001', '002', '003', '004'] as `output` value
test_workflow = Workflow(
    base_dir = '/path/to/base/dir',
    name = 'test_workflow'
    )
test_workflow.connect([
	# node_1 will receive the evaluation of :
	# 	list_intersection(['001', '002', '003', '004'], ['002', '004', '005'])
	#	as in_value
    (node_0, node_1, [(('output', list_intersection, ['002', '004', '005']), 'in_value')])
    ])
test_workflow.run()
```

> [!TIP]
> Use a [nipype.MapNode](https://nipype.readthedocs.io/en/latest/api/generated/nipype.pipeline.engine.nodes.html#nipype.pipeline.engine.nodes.MapNode) to run these functions on lists instead of unitary contents. E.g.: the `remove_file` function of `narps_open.core.common` only removes one file at a time, but feel free to pass a list of files using a `nipype.MapNode`.

```python
from nipype import MapNode, Function
from narps_open.core.common import remove_file

# Create the MapNode so that the `remove_file` function handles lists of files
remove_files_node = MapNode(Function(
    function = remove_file,
    input_names = ['_', 'file_name'],
    output_names = []
    ), name = 'remove_files_node', iterfield = ['file_name'])

# ... A couple of lines later, in the Worlflow definition
test_workflow = Workflow(base_dir = '/home/bclenet/dev/tests/nipype_merge/', name = 'test_workflow')
test_workflow.connect([
	# ...
	# Here we assume the select_node's output `out_files` is a list of files
    (select_node, remove_files_node, [('out_files', 'file_name')])
	# ...
    ])
```

## narps_open.core.common

This module contains a set of functions that nearly every pipeline could use.

* `remove_file` remove a file when it is not needed anymore (to save disk space)

```python
from narps_open.core.common import remove_file

# Remove the /path/to/the/image.nii.gz file
remove_file('/path/to/the/image.nii.gz')
```

* `elements_in_string` : return the first input parameter if it contains one element of second parameter (None otherwise).

```python
from narps_open.core.common import elements_in_string

# Here we test if the file 'sub-001_file.nii.gz' belongs to a group of subjects.
elements_in_string('sub-001_file.nii.gz', ['005', '006', '007']) # Returns None
elements_in_string('sub-001_file.nii.gz', ['001', '002', '003']) # Returns 'sub-001_file.nii.gz'
```

> [!TIP]
> This can be generalised to a group of files, using a `nipype.MapNode`!

* `clean_list` : remove elements of the first input parameter (list) if it is equal to the second parameter.

```python
from narps_open.core.common import clean_list

# Here we remove subject 002 from a group of subjects.
clean_list(['002', '005', '006', '007'], '002')
```

* `list_intersection` : return the intersection of two lists.

```python
from narps_open.core.common import list_intersection

# Here we keep only subjects that are in the equalRange group and selected for the analysis.
equal_range_group = ['002', '004', '006', '008']
selected_for_analysis = ['002', '006', '010']
list_intersection(equal_range_group, selected_for_analysis) # Returns ['002', '006']
```

## narps_open.core.image

This module contains a set of functions dedicated to computations on images.

 * `get_voxel_dimensions` : returns the voxel dimensions of an image

```python
# Get dimensions of voxels along x, y, and z in mm (returns e.g.: [1.0, 1.0, 1.0]).
get_voxel_dimensions('/path/to/the/image.nii.gz')
```
