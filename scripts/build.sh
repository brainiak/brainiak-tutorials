#!/usr/bin/env bash

# Verify that we are on the master branch
if [[ $(git branch | grep \* | cut -d ' ' -f2) != "master" ]]; then
  echo "ERROR: Please build from master branch"
  exit
fi

# Verify that we are on a clean master branch
if [[ -n $(git status -s) ]]; then
  echo "ERROR: Please build from clean branch (i.e., no untracked or uncommitted files)"
  exit
fi

# Pull any updates
git pull origin master

if [[ -d brainiak ]]; then
  pushd brainiak

  # Verify that we are on the master branch
  if [[ $(git branch | grep \* | cut -d ' ' -f2) != "master" ]]; then
    echo "ERROR: Please build from master branch"
    exit
  fi

  # Verify that we are on a clean master branch
  if [[ -n $(git status -s) ]]; then
    echo "ERROR: Please build from clean branch (i.e., no untracked or uncommitted files)"
    exit
  fi

  git pull origin master
  popd
else
  git clone https://github.com/brainiak/brainiak
fi

pushd brainiak
BRAINIAK_HASH=$(git rev-parse --short HEAD)
popd

TUTORIALS_HASH=$(git rev-parse --short HEAD)

echo "brainiak git hash: $BRAINIAK_HASH"
echo "tutorials git hash: $TUTORIALS_HASH"

TAG=$(date +%Y%m%d)-0-$TUTORIALS_HASH-$BRAINIAK_HASH

echo "Creating new docker image with tag" brainiak/tutorials:$TAG

docker build -t brainiak/tutorials:$TAG .
docker push brainiak/tutorials:$TAG

docker tag brainiak/tutorials:$TAG brainiak/tutorials:latest
docker push brainiak/tutorials:latest
