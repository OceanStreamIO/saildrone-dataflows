#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo ".env file not found!"
  exit 1
fi

# Pass all parameters to dask-worker
dask worker tcp://192.168.0.17:8786 "$@"
