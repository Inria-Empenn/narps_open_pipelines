#! /bin/bash

set -ex -o pipefail

# used to push the Docker image for a project in circle CI

if [[ -n "${DOCKER_TOKEN}" ]]; then

    # make sure we have a lowercase repo
    user_name="Inria-Empenn"
    repo_name=$(echo "${CIRCLE_PROJECT_REPONAME}" | tr '[:upper:]' '[:lower:]')

    echo "${DOCKER_TOKEN}" | docker login -u "${DOCKER_USER}" --password-stdin

    : "Pushing to DockerHub ${user_name}/${repo_name}:unstable"
    docker tag "${user_name}/${repo_name}" "${user_name}/${repo_name}:unstable"
    docker push "${user_name}/${repo_name}:unstable"

    if [[ -n "${CIRCLE_TAG}" ]]; then
        : "Pushing to DockerHub ${user_name}/${repo_name}:${CIRCLE_TAG}"
        docker push "${user_name}/${repo_name}:latest"
        docker tag "${user_name}/${repo_name}" "${user_name}/${repo_name}:${CIRCLE_TAG}"
        docker push "${user_name}/${repo_name}:${CIRCLE_TAG}"
    fi

else
    : "No DOCKER_TOKEN, skipping push to DockerHub"
    exit 1
fi
