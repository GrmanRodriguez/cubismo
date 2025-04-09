#!/bin/bash

if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    set -o allexport
    source .env
    set +o allexport
else
    echo "No .env file found."
    exit 1
fi

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "CHECKPOINT_PATH is not set in the environment."
    exit 1
fi

LOCAL_CHECKPOINT_PATH="./checkpoint.ckpt"
cp $CHECKPOINT_PATH $LOCAL_CHECKPOINT_PATH

GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null)

if [ -z "$GIT_COMMIT" ]; then
    echo "Not a git repository or git not available."
    exit 1
fi

# Extract relevant checkpoint data from the filename
BASE_NAME=$(basename "$CHECKPOINT_PATH")
PARENT_DIR=$(basename "$(dirname "$CHECKPOINT_PATH")")

if [[ "$BASE_NAME" =~ epoch=([0-9]+) ]]; then
    EPOCH_NUM="${BASH_REMATCH[1]}"
    IMAGE_TAG="${PARENT_DIR}-epoch_${EPOCH_NUM}"
else
    # If checkpoint name doesn't match convention then use just filename
    IMAGE_TAG="${BASE_NAME%.*}"
fi

IMAGE_TAG="${GIT_COMMIT}-${IMAGE_TAG}"

DOCKER_BUILDKIT=1 docker build --build-arg CHECKPOINT_PATH=$LOCAL_CHECKPOINT_PATH -t cubismo:${IMAGE_TAG} -f docker/Dockerfile .

rm $LOCAL_CHECKPOINT_PATH
