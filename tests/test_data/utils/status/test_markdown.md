# Work progress status for each pipeline
The *status* column tells whether the work on the pipeline is :
<br>:red_circle: not started yet
<br>:orange_circle: in progress
<br>:green_circle: completed
<br><br>The *main software* column gives a simplified version of what can be found in the team descriptions under the `general.software` column.
<br><br>The *reproducibility* column rates the pipeline as follows:
 * default score is :star::star::star::star:;
 * -1 if the team did not use fmriprep data;
 * -1 if the team used several pieces of software (e.g.: FSL and AFNI);
 * -1 if the team used custom or marginal software (i.e.: something else than SPM, FSL, AFNI or nistats);
 * -1 if the team did not provided his source code.

| team_id | status | main software | fmriprep used ? | related issues | related pull requests | excluded from NARPS analysis | reproducibility |
| --- |:---:| --- | --- | --- | --- | --- | --- |
| Q6O0 | :green_circle: | SPM | Yes |  |  | No | :star::star::star::black_small_square:<br /> |
| UK24 | :orange_circle: | SPM | No | [2](url_issue_2),  |  | No | :star::star::black_small_square::black_small_square:<br /> |
| 2T6S | :orange_circle: | SPM | Yes | [5](url_issue_5),  | [3](url_pull_3),  | No | :star::star::star::black_small_square:<br /> |
| 1KB2 | :red_circle: | FSL | No |  |  | No | :star::star::black_small_square::black_small_square:<br /> |
| C88N | :red_circle: | SPM | Yes |  |  | No | :star::star::star::black_small_square:<br /> |
