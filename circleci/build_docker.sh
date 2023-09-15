#! /bin/bash

set -eux -o pipefail

# used to build the Docker image for a project in circle CI
#
#  assumes that the docker has been cached in ${HOME}/docker/image.tar
#
# - save_cache:
#     key: my_cache
#     paths:
#     - ~/docker
#     - ~/data
#

# make sure we have a lowercase repo
user_name="Inria-Empenn"
repo_name=$(echo "${CIRCLE_PROJECT_REPONAME}" | tr '[:upper:]' '[:lower:]')

if [[ -e "${HOME}/docker/image.tar" ]]; then
    docker load -i "${HOME}/docker/image.tar"
fi
git describe --tags --always > version
docker build -t "${user_name}/${repo_name}" .
mkdir -p "${HOME}/docker"
docker save "${user_name}/${repo_name}" > "${HOME}/docker/image.tar"
docker images
