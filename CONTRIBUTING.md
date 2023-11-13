# How to contribute to NARPS Open Pipelines ? 

For the reproductions, we are especially looking for contributors with the following profiles:
 - 👩‍🎤 SPM, FSL, AFNI or nistats has no secrets for you? You know this fMRI analysis software by heart 💓. Please help us by reproducing the corresponding NARPS pipelines. 👣 after step 1, follow the fMRI expert trail.
 - 🧑‍🎤 You are a nipype guru? 👣 after step 1, follow the nipype expert trail.

# Step 1: Choose a pipeline to reproduce :keyboard:
:thinking: Not sure which pipeline to start with ? 🚦The [pipeline dashboard](https://github.com/Inria-Empenn/narps_open_pipelines/wiki/pipeline_status) provides the progress status for each pipeline. You can pick any pipeline that is in red (not started).

Need more information to make a decision? The `narps_open.utils.description` module of the project, as described [in the documentation](/docs/description.md) provides easy access to all the info we have on each pipeline.

When you are ready, [start an issue](https://github.com/Inria-Empenn/narps_open_pipelines/issues/new/choose) and choose **Pipeline reproduction**!

# Step 2: Reproduction 

## 🧑‍🎤 NiPype trail

We created templates with modifications to make and holes to fill to create a pipeline. You can find them in [`narps_open/pipelines/templates`](/narps_open/pipelines/templates).

If you feel it could be better explained, do not hesitate to suggest modifications for the templates.

Feel free to have a look to the following pipelines, these are examples :
| team_id | softwares | fmriprep used ? | pipeline file |
| --- | --- | --- | --- |
| 2T6S | SPM | Yes | [/narps_open/pipelines/team_2T6S.py](/narps_open/pipelines/team_2T6S.py) |
| X19V | FSL | Yes | [/narps_open/pipelines/team_X19V.py](/narps_open/pipelines/team_2T6S.py) |

## 👩‍🎤 fMRI software trail

...

## Find or propose an issue :clipboard:
Issues are very important for this project. If you want to contribute, you can either **comment an existing issue** or **proposing a new issue**. 

### Answering an existing issue :label:
To answer an existing issue, make a new comment with the following information: 
  - Your name and/or github username
  - The step you want to contribute to 
  - The approximate time needed 

### Proposing a new issue :bulb:
In order to start a new issue, click [here](https://github.com/Inria-Empenn/narps_open_pipelines/issues/new/choose) and choose the type of issue you want:
  - **Feature request** if you aim at improving the project with your ideas ;
  - **Bug report** if you encounter a problem or identified a bug ;
  - **Classic issue** to ask question, give feedbacks...

Some issues are (probably) already open, please browse them before starting a new one. If your issue was already reported, you may want complete it with details or other circumstances in which a problem appear. 

## Pull Requests :inbox_tray:
Pull requests are the best way to get your ideas into this repository and to solve the problems as fast as possible.

### Make A Branch :deciduous_tree:
Create a separate branch for each issue you're working on. Do not make changes to the default branch (e.g. master, develop) of your fork.

### Push Your Code :outbox_tray:
Push your code as soon as possible.

### Create the Pull Request (PR) :inbox_tray:
Once you pushed your first lines of code to the branch in your fork, visit [this page](https://github.com/Inria-Empenn/narps_open_pipelines/pulls) to start creating a PR for the NARPS Open Pipelines project.

:warning: Please create a **Draft Pull Request** as described [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork), and please stick to the PR description template as much as possible.

Continue writing your code and push to the same branch. Make sure you perform all the items of the PR checklist.

### Request Review :disguised_face:
Once your PR is ready, you may add a reviewer to your PR, as described [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/requesting-a-pull-request-review) in the GitHub documentation.

Please turn your Draft Pull Request into a "regular" Pull Request, by clicking **Ready for review** in the Pull Request page.

**:wave: Thank you in advance for contributing to the project!**

## Additional resources

 - git and Gitub: general guidelines can be found [here](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) in the GitHub documentation. 
