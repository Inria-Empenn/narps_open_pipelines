# last update: June 2023

from nipype import Node, Function, Workflow,IdentityInterface
import subprocess
from nipype.interfaces.io import SelectFiles, DataSink
from os.path import abspath
from os.path import join as opj
import os

# define environment
data_dir = "/home/jlefortb/narps_open_pipelines/data/original/ds001734/"
results_dir =  "/home/jlefortb/narps_open_pipelines/data/results/"
working_dir = "/home/jlefortb/narps_open_pipelines/"

# define subject ids
dir_list = os.listdir(data_dir)
# Subject list (to which we will do the analysis)
subject_list = []
for dirs in dir_list:
    if dirs[0:3] == 'sub':
        subject_list.append(dirs[-3:])


# Infosource Node - To iterate on subjects + get directoris paths
infosource = Node(interface=IdentityInterface(fields = ['subject_id', 'data_dir', 'results_dir', 'working_dir'],
                    data_dir = data_dir,
                    results_dir = results_dir,
                    working_dir = working_dir),
                  name = 'infosource')

infosource.iterables = [('subject_id', subject_list)]


templates = {'command': abspath('narps_open/pipelines/team_6VV2_AfniProc.simple')}
# Create SelectFiles node
files = Node(SelectFiles(templates),
          name='files')
# Location of the dataset folder
files.inputs.base_directory = '.'

# create stimuli file the afni way
def create_stimuli_file(subject, data_dir):
    # create 1D stimuli file :
    import pandas as pd 
    from os.path import join as opj
    df_run1 = pd.read_csv(opj(data_dir, "sub-{}/func/sub-{}_task-MGT_run-01_events.tsv".format(subject, subject)), sep="\t")
    df_run1 = df_run1[["onset", "gain", "loss"]].T
    df_run2 = pd.read_csv(opj(data_dir, "sub-{}/func/sub-{}_task-MGT_run-02_events.tsv".format(subject, subject)), sep="\t")
    df_run2 = df_run2[["onset", "gain", "loss"]].T
    df_run3 = pd.read_csv(opj(data_dir, "sub-{}/func/sub-{}_task-MGT_run-03_events.tsv".format(subject, subject)), sep="\t")
    df_run3 = df_run3[["onset", "gain", "loss"]].T
    df_run4 = pd.read_csv(opj(data_dir, "sub-{}/func/sub-{}_task-MGT_run-04_events.tsv".format(subject, subject)), sep="\t")
    df_run4 = df_run4[["onset", "gain", "loss"]].T

    df_gain = pd.DataFrame(index=range(0,4), columns=range(0,64))
    df_gain.loc[0] = ["{}*{}".format(df_run1[col].loc['onset'], df_run1[col].loc['gain']) for col in range(0, 64)]
    df_gain.loc[1] = ["{}*{}".format(df_run2[col].loc['onset'], df_run2[col].loc['gain']) for col in range(0, 64)]
    df_gain.loc[2] = ["{}*{}".format(df_run3[col].loc['onset'], df_run3[col].loc['gain']) for col in range(0, 64)]
    df_gain.loc[3] = ["{}*{}".format(df_run4[col].loc['onset'], df_run4[col].loc['gain']) for col in range(0, 64)]
    df_loss = pd.DataFrame(index=range(0,4), columns=range(0,64))
    df_loss.loc[0] = ["{}*{}".format(df_run1[col].loc['onset'], df_run1[col].loc['loss']) for col in range(0, 64)]
    df_loss.loc[1] = ["{}*{}".format(df_run2[col].loc['onset'], df_run2[col].loc['loss']) for col in range(0, 64)]
    df_loss.loc[2] = ["{}*{}".format(df_run3[col].loc['onset'], df_run3[col].loc['loss']) for col in range(0, 64)]
    df_loss.loc[3] = ["{}*{}".format(df_run4[col].loc['onset'], df_run4[col].loc['loss']) for col in range(0, 64)]

    df_gain.to_csv(opj(data_dir, "sub-{}/func/times+gain.1D".format(subject)), 
            sep='\t', index=False, header=False)
    df_loss.to_csv(opj(data_dir, "sub-{}/func/times+loss.1D".format(subject)), 
            sep='\t', index=False, header=False)
    print("Done")

create_stimuli = Node(Function(input_names=["subject", "data_dir"],
                                    output_names=["Stimuli"],
                                    function=create_stimuli_file),
              name='create_stimuli')


# launch afni bash script
def run(command, results_dir, subject, data_dir):
    import subprocess
    subject= "sub-{}".format(subject)
    subprocess.run([command, results_dir, subject, data_dir])
    print("Done")

afni_proc = Node(Function(input_names=["command", "results_dir", "subject", "data_dir"],
                                    output_names=["Adni_proc_ran"],
                                    function=run),
              name='afni_proc')



####### build workflow
wf_run = Workflow(base_dir = opj(results_dir, working_dir), name="Afni_proc_through_nipype")
wf_run.base_dir = '.'
wf_run.connect([(infosource, create_stimuli, [('subject_id', 'subject')]),
                (infosource, create_stimuli, [("data_dir", "data_dir")]),
                (infosource, afni_proc, [('subject_id', 'subject')]),
                (infosource, afni_proc, [("results_dir", "results_dir")]),
                (infosource, afni_proc, [("data_dir", "data_dir")]),
                (files, afni_proc, [("command", "command")])
                ])
# wf_run.write_graph()
wf_run.run()