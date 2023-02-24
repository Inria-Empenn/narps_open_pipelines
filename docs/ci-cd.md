# :package: Continuous Integration (CI) and Continuous Deployment (CD) for the NARPS open pipelines project

:mega: This file descripes how CI/CD works for the project.

## :octopus: CI on GitHub

GitHub allows to launch CI workflows using [Actions](https://docs.github.com/en/actions).

See GitHub's documentation on [worflow syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions) to write your own workflows.

These worflows are YAML files, located in the `.github/workflows/` directory.

### CI scheme

The following scheme shows the currently set up CI workflows on GitHub, their location for running, as well as their trigger.

![Scheme of CI for NARPS open pipelines](/docs/assets/ci-scheme.svg)

### Dedicated runners

Because the testing and running NARPS open pipelines highly rely on large amount of data (subject images, teams results) and sizeable Docker images, dedicated [self-hosted runners](https://docs.github.com/en/actions/hosting-your-own-runners/about-self-hosted-runners) were assigned to the project by [Inria-Empenn](https://github.com/Inria-Empenn). They ensure CI jobs are run in a proper environment, with all the data at hand.

### Workflow triggers

Most workflows are triggered by pull requests (PR) because their purpose is to test code coming from contributors.
Some workflows are triggered by pushes to allow contributors to test their code before creating pull requests.
Because the `main` branch is supposed to be safe and stable, no testing workflows are triggered when pushing to this branch.

Furthermore, you can always [prevent GitHub from running workflows](https://docs.github.com/en/actions/managing-workflow-runs/skipping-workflow-runs) by adding a `[skip ci]` in your commit message.

### CI workflows

For now, the following workflows are set up:

* [code_quality.yml](/.github/workflows/code_quality.yml)
	* **What does it do ?** It performs a static analysis of the python code (see the [testing](/docs/testing.md) topic of the documentation for more information).
	* **When is it launched ?** For every push (except ) if there are changes on `.py` files.
	* **Where does it run ?** On GitHub servers.
	* **How can I see the results ?** Outputs (logs of pylint) are stored as [downloadable artifacts](https://docs.github.com/en/actions/managing-workflow-runs/downloading-workflow-artifacts) during 15 days after the push.

* [pipelines.yml](/.github/workflows/pipelines.yml)
	* **What does it do ?** It runs all the tests for changed pipelines.
	* **When is it launched ?** For every pull request, if a pipeline file changed inside `narps_open/pipelines/`.
	* **Where does it run ?** On Empenn runners.
	* **How can I see the results ?** Outputs (logs of pytest) are stored as downloadable artifacts during 15 days after the push.

* [tests.yml](/.github/workflows/tests.yml)
	* **What does it do ?** It runs all the changed tests for the project.
	* **When is it launched ?** For every pull request, if a test file changed inside `tests/`.
	* **Where does it run ?** On Empenn runners.
	* **How can I see the results ?** Outputs (logs of pytest) are stored as downloadable artifacts during 15 days after the push.

* [unit_testing.yml](/.github/workflows/unit_testing.yml)
	* **What does it do ?** It runs all the unit tests for the project (see the [testing](/docs/testing.md) topic of the documentation for more information).
	* **When is it launched ?** For every pull request, if a file changed inside `narps_open/` other than in `narps_open/pipeline`.
	* **Where does it run ?** On Empenn runners.
	* **How can I see the results ?** Outputs (logs of pytest) are stored as downloadable artifacts during 15 days after the push.

### Security

We take these precautions following [GitHub good security practices](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions), and in order to avoid malicious code to be run through GitHub Actions workflows.

- The [CODEOWNERS](/.github/CODEOWNERS) file assigns owners for `.github/workflows` and `.gitlab-ci.yml` (see [CODEOWNERS reference](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)). This prevents unwanted changes to these files that could alter the CI-CD workflows integrity.
- Workflows triggered by pull requests from forks of the repository are submitted to approval, as described in GitHub's documentation about [managing actions for a public repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository#controlling-changes-from-forks-to-workflows-in-public-repositories)

## :fox_face: CI on GitLab

A `.gitlab-ci.yml` file is provided in the repository as an example for contributors who would like to trigger [GitLab CI](https://docs.gitlab.com/ee/ci/) scripts from a [GitLab mirror repository](https://docs.gitlab.com/ee/user/project/repository/mirror/) of their fork of Inria-Empenn/narps_open_pipelines.
