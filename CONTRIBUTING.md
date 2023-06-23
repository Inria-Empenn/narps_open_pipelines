# How to contribute to NARPS Open Pipelines ? 

General guidelines can be found [here](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) in the GitHub documentation. 

## Reproduce a pipeline :keyboard:
:thinking: Not sure which one to start with ? You can have a look on [this table](https://github.com/Inria-Empenn/narps_open_pipelines/wiki/pipeline_status) giving the work progress status for each pipeline. This will help choosing the one that best suits you!

Need more information ? You can have a look to the pipeline decription [here](https://docs.google.com/spreadsheets/d/1FU_F6kdxOD4PRQDIHXGHS4zTi_jEVaUqY_Zwg0z6S64/edit?usp=sharing). Also feel free to use the description module of the project, as described [in the documentation](/docs/description.md).

When you are ready, [start an issue](https://github.com/Inria-Empenn/narps_open_pipelines/issues/new/choose) and choose **Pipeline reproduction**!

### If you have experience with NiPype
We created a template with modifications to make and holes to fill to create a pipeline. You can find it on [`narps_open/pipelines/pipeline_template.py`](/narps_open/pipelines/pipeline_template.py). 
If you feel it could be better explained, do not hesitate to modify the template or to create an issue about it.

### If you have experience with the original software package but not with NiPype
A fantastic tool named [Giraffe](https://giraffe.tools/porcupine/TimVanMourik/GiraffePlayground/master) is available. It allows you to create a graph of your pipeline using NiPype functions but without coding! Just save your NiPype script in a .py file and send it as a new issue, we will convert this script to a script which works with our specific parameters. 

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
Once you pushed your first lines of code to the branch in your fork, visit [this page](https://github.com/Inria-Empenn/narps_open_pipelines/pulls) to start creating a PR for the NARPS Open Pipelines project. :warning: Please create a **Draft Pull Request** as described [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) in the GitHub documentation.

Continue writing your code and push to the same branch. Make sure to respect all the items of the PR checklist.

### Request Review :disguised_face:
Once your PR is ready, you may add a reviewer to your PR, as described [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/requesting-a-pull-request-review) in the GitHub documentation.

Please turn your Draft Pull Request into a "regular" Pull Request, by clicking **Ready for review** in the Pull Request page.

**:wave: Thank you in advance for contributing to the project!**
