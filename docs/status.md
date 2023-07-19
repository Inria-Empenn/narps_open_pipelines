# Access the work progress status pipelines

The class `PipelineStatusReport` of module `narps_open.utils.status` allows to create a report containing the following information for each pipeline:
* the software it uses (collected from the `categorized_for_analysis.analysis_SW` of the [team description](/docs/description.md)) ;
* whether it uses data from fMRIprep or not ;
* a list of issues related to it (the opened issues of the project that have the team ID inside their title or description) ;
* a work progress status : `idle`, `progress`, or `done`.

This allows contributors to best select the pipeline they want/need to contribute to. For this purpose, the GitHub Actions workflow [`.github/workflows/pipeline_status.yml`](/.github/workflows/pipeline_status.yml) allows to dynamically generate the report and to store it in the [project's wiki](https://github.com/Inria-Empenn/narps_open_pipelines/wiki).

The class allows to output the report in a JSON format or in a Markdown format.

Here is an example on how to use it:

```python
from narps_open.utils.status import PipelineStatusReport

# Create and generate the report
report = PipelineStatusReport()
report.generate()

# Access the contents of the report
pipeline_info = report.contents['2T6S']
print(pipeline_info['softwares'])
print(pipeline_info['softwares'])
print(pipeline_info['fmriprep'])
print(pipeline_info['issues'])
print(pipeline_info['status'])

# Create a nice Markdown formatting for the report
report.markdown() # Returns a string containing the markdown
```

You can also use the command-line tool as so. Option `-t` is for the team id, option `-d` allows to print only one of the sub parts of the description among : `general`, `exclusions`, `preprocessing`, `analysis`, and `categorized_for_analysis`.

```bash
python narps_open/utils/status -h
# usage: status.py [-h] [--json | --md]
# 
# Get a work progress status report for pipelines.
# 
# options:
#   -h, --help  show this help message and exit
#   --json      output the report as JSON
#   --md        output the report as Markdown

python narps_open/utils/status --json
# {
#     "08MQ": {
#         "softwares": "FSL",
#         "fmriprep": "No",
#         "issues": {},
#         "status": "idle"
#     },
#     "0C7Q": {
#         "softwares": "FSL, AFNI",
#         "fmriprep": "Yes",
#         "issues": {},
#         "status": "idle"
#     },
# ...

python narps_open/utils/status --md
# | team_id | status | softwares used | fmriprep used ? | related issues |
# | --- |:---:| --- | --- | --- |
# | 08MQ | :red_circle: | FSL | No |  |
# | 0C7Q | :red_circle: | FSL, AFNI | Yes |  |
# | 0ED6 | :red_circle: | SPM | No |  |
# | 0H5E | :red_circle: | SPM | No |  |
# ...
```
