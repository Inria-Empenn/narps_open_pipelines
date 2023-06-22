
# last update: June 2023

from nipype import Node, Function, Workflow,IdentityInterface
import subprocess
from nipype.interfaces.io import SelectFiles, DataSink
from os.path import abspath
from os.path import join as opj

# selectfile
templates = {'command_sub1': abspath('outputs_6VV2_test_nipype/6VV2_AfniProc.simple')}
# Create SelectFiles node
files = Node(SelectFiles(templates),
          name='files')
# Location of the dataset folder
files.inputs.base_directory = '.'

# launch afni bash script
def run(command):
	import subprocess
	subprocess.run([command])
	print("Done")

afni_proc = Node(Function(input_names=["command"],
                                    output_names=["Adni_proc_ran"],
                                    function=run),
              name='afni_proc')

####### build workflow
wf_run = Workflow(name="Afni_proc_through_nipype")
wf_run.base_dir = '.'
wf_run.connect([(files, afni_proc, [("command_sub1", "command")])
                ])
wf_run.run()