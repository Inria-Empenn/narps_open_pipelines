# How to contribute to NARPS Open Pipelines ? 

For the reproductions, we are especially looking for contributors with the following profiles:
 - `🧠 fMRI soft` SPM, FSL, AFNI or nistats has no secrets for you ; you know one of these fMRI analysis tools by :heart:.
 - `🐍 Python` You are a Python guru, willing to use [Nipype](https://nipype.readthedocs.io/en/latest/).

In the following, read the instruction sections where the badge corresponding to your profile appears.

## 1 - Choose a pipeline
`🧠 fMRI soft` `🐍 Python`

Not sure which pipeline to start with :thinking:? The [pipeline dashboard](https://github.com/Inria-Empenn/narps_open_pipelines/wiki/pipeline_status) provides the progress status for each pipeline. You can pick a pipeline that is not fully reproduced, i.e.: not started :red_circle: or in progress :orange_circle: . Also have a look to the [pipeline reproduction management page](https://github.com/orgs/Inria-Empenn/projects/1/views/1) in order to get in touch with contributors working on the same pipeline.

> [!NOTE]
> Need more information to make a decision? The `narps_open.utils.description` module of the project, as described [in the documentation](/docs/description.md) provides easy access to all the info we have on each pipeline.

## 2 - Interact using issues
`🧠 fMRI soft` `🐍 Python`

Browse [issues](https://github.com/Inria-Empenn/narps_open_pipelines/issues/) before starting a new one. If the pipeline is :orange_circle:, the associated issues are listed on the [pipeline dashboard](https://github.com/Inria-Empenn/narps_open_pipelines/wiki/pipeline_status).

You can either:
* comment on an existing issue with details or your findings about the pipeline;
* [start an issue](https://github.com/Inria-Empenn/narps_open_pipelines/issues/new/choose) and choose **Pipeline reproduction**.

> [!WARNING]
> As soon as the issue is marked as `🏁 status: ready for dev` you can proceed to the next step.

## 3 - Use pull requests
`🐍 Python`

1. [Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the repository;
2. create a separate branch for the issue you're working on (do not make changes to the default branch of your fork).
3. push your work to the branch as soon as possible;
4. visit [this page](https://github.com/Inria-Empenn/narps_open_pipelines/pulls) to start a draft pull request.

> [!WARNING]
> Make sure you create a **Draft Pull Request** as described [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork), and please stick to the description of the pull request template as much as possible.

## 4 - Reproduce pipeline

### Translate the pipeline description into code
`🐍 Python`

Write your code and push it to the branch. Make sure you perform all the items of the pull request checklist.

From the description provided by the team you chose, write Nipype workflows that match the steps performed by the teams (preprocessing, run level analysis, subject level analysis, group level analysis).

We created templates with modifications to make and holes to fill to help you with that. Find them in [`narps_open/pipelines/templates`](/narps_open/pipelines/templates).

> [!TIP]
> Have a look to the already reproduced pipelines, as examples :
> | team_id | softwares | fmriprep used ? | pipeline file |
> | --- | --- | --- | --- |
> | Q6O0 | SPM | Yes | [/narps_open/pipelines/team_Q6O0.py](/narps_open/pipelines/team_Q6O0.py) |

Once your work is ready, you may ask a reviewer to your pull request, as described [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/requesting-a-pull-request-review). Please turn your draft pull request into a *regular* pull request, by clicking **Ready for review** in the pull request page.

### Run the pipeline and produce evidences
`🧠 fMRI soft`

From the description provided by the team you chose, perform the analysis on the associated software to get as many metadata (log, configuration files, and other relevant files for reproducibility) as possible from the execution. Complementary hints and comments on the process would definitely be welcome, to enrich the description (e.g.: relevant parameters not written in the description, etc.).

Especially these files contain valuable information about model design:
* for FSL pipelines, `design.fsf` setup files coming from FEAT ;
* for SPM pipelines, `matlabbatch` files.

You can attach these files as comments on the pipeline reproduction issue.

**:wave: Thank you for contributing to the project!**
