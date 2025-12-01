#!/bin/bash
# Wrapper script to run training from project root

cd "$(dirname "$0")/src" && python train.py "$@"
