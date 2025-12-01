#!/bin/bash
# Wrapper script to run prediction from project root

cd "$(dirname "$0")/src" && python predict.py "$@"
