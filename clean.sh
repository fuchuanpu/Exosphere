#!/usr/bin/env bash

BASE_NAME=$(basename $(pwd))

if [ $BASE_NAME != "ExoSphere_Demo" ] && [ $BASE_NAME != "exosphere_demo" ]; then
    echo "This script should be executed in the root directory."
    exit -1
fi

function clean_dir () {
    if [ $# != 2 ]; then
        echo "Invalid number of arguments."
        return -1;
    fi

    if [ -d "./$1" ]; then
        cd "./$1"
        TARGET_CTR=$(ls -al | grep .$2 | wc -l)
        if (( $((TARGET_CTR)) > 0 )); then
            rm *.$2
        fi
        cd -
    fi
    return 0;
}

clean_dir log log
clean_dir figure png
