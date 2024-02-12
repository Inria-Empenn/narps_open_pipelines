# Access the work progress status pipelines

The class `PipelineStatusReport` of module `narps_open.utils.status` allows to create a report containing the following information for each pipeline:
* a work progress status : `idle`, `progress`, or `done`;
* the software it uses (collected from the `categorized_for_analysis.analysis_SW` of the [team description](/docs/description.md)) ;
* whether it uses data from fMRIprep or not ;
* a list of issues related to it (the opened issues of the project that have the team ID inside their title or description) ;
* a list of pull requests related to it (the opened pull requests of the project that have the team ID inside their title or description) ;
* whether it was excluded from the original NARPS analysis ;
* a reproducibility rating :
  * default score is 4;
  * -1 if the team did not use fmriprep data;
  * -1 if the team used several pieces of software (e.g.: FSL and AFNI);
  * -1 if the team used custom or marginal software (i.e.: something else than SPM, FSL, AFNI or nistats);
  * -1 if the team did not provided his source code.

This allows contributors to best select the pipeline they want/need to contribute to. For this purpose, the GitHub Actions workflow [`.github/workflows/pipeline_status.yml`](/.github/workflows/pipeline_status.yml) allows to dynamically generate the report and to store it in the [project's wiki](https://github.com/Inria-Empenn/narps_open_pipelines/wiki).

Pipelines are sorted by the following columns :
1. status
2. software used
3. fmriprep used?

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

You can also use the command-line tool as so.

> [!TIP]
> In the following examples, use `narps_open_status` or `python narps_open/utils/status.py` indifferently to launch the command line tool.

```bash
narps_open_status -h
# usage: status.py [-h] [--json | --md]
# 
# Get a work progress status report for pipelines.
# 
# options:
#   -h, --help  show this help message and exit
#   --json      output the report as JSON
#   --md        output the report as Markdown

narps_open_status --json
# {
#     "08MQ": {
#         "softwares": "FSL",
#         "fmriprep": "No",
#         "issues": {},
#         "excluded": "No",
#         "reproducibility": 3,
#         "reproducibility_comment": "",
#         "issues": {},
#         "pulls": {},
#         "status": "2-idle"
#     },
#     "0C7Q": {
#         "softwares": "FSL, AFNI",
#         "fmriprep": "Yes",
#         "issues": {},
#         "excluded": "No",
#         "reproducibility": 3,
#         "reproducibility_comment": "",
#         "issues": {},
#         "pulls": {},
#         "status": "idle"
#     },
# ...

narps_open_status --md
# ...
# | team_id | status | main software | fmriprep used ? | related issues | related pull requests | excluded from NARPS analysis | reproducibility |
# | --- |:---:| --- | --- | --- | --- | --- | --- |
# | Q6O0 | :green_circle: | SPM | Yes |  |  | No | :star::star::star::black_small_square:<br /> |
# | UK24 | :orange_circle: | SPM | No | [2](url_issue_2),  |  | No | :star::star::black_small_square::black_small_square:<br /> |
# ...
```
