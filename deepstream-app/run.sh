#!/bin/bash
FILE=mounted_config.txt
if test -f "$FILE"; then
    echo "Run mounted config ..."
    ./deepstream-app -c mounted_config.txt
else
    echo "Run default config ..."
    ./deepstream-app -c deepstream_app_config_pose.txt
fi