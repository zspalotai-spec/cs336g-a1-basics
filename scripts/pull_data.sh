#!/bin/bash
# Pushes into $PROJECT_ROOT/data and downloads necessary train/val files.

# Project directory structure
readonly PROJECT_ROOT="${PWD}/.."
readonly DATA_SUBDIR="${PROJECT_ROOT}/data"

# Data files
readonly TINYSTORIES_TRAIN="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
readonly TINYSTORIES_VAL="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
readonly OPEN_WEB_TEXT_TRAIN="https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz"
readonly OPEN_WEB_TEXT_VAL="https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz"

# Helper around print statements.
function print_info() {

  function print_sep() {
    echo "$(seq -s'=' 0 "$(( $(tput cols) - 1))" | tr -d '[:digit:]')"
  }

  print_sep
  echo "$1"
  print_sep
}

function main() {
    mkdir -p "${DATA_SUBDIR}"
    pushd "${DATA_SUBDIR}" > /dev/null

    # TinyStories
    print_info "Pulling TinyStories data..."
    curl -LO "${TINYSTORIES_TRAIN}"
    curl -LO "${TINYSTORIES_VAL}"

    # OWT
    print_info "Pulling Open Web Text data..."
    curl -LO "${OPEN_WEB_TEXT_TRAIN}"
    gunzip "$(basename ${OPEN_WEB_TEXT_TRAIN})"
    curl -LO "${OPEN_WEB_TEXT_VAL}"
    gunzip "$(basename ${OPEN_WEB_TEXT_VAL})"

    popd > /dev/null
}

main "$@"
