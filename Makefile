#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Make FallDetection Dataset
dataset_fd: download_fd preprocess_fd

## Download FD dataset
download_fd:
    cd data/raw/FD; ./download_fd.sh

## Preprocess FD dataset
preprocess_fd: download_fd
    cd data/raw/FD; # Something here