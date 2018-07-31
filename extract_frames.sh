#!/bin/bash
set -e

for f in /DATACENTER/sshfs/ActivityNet-v1.3/videos/*.mp4
do
    filename=$(basename $f)
    if [ -d /DATACENTER/2/xue/SRC_FOLDER/frames/${filename%.*} ]; then
        continue 
    fi
    echo $f
    mkdir /DATACENTER/2/xue/SRC_FOLDER/frames/${filename%.*}
#    chmod +777 frames/${filename%.*}
    ffmpeg -i $f -vf scale=-1:240 /DATACENTER/2/xue/SRC_FOLDER/frames/${filename%.*}/img_%5d.jpg -hide_banner -loglevel error >>error.log
done
