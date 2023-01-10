#!/bin/bash
set -e

DATA_DIR=$1

if [ ! -d $DATA_DIR ]; then
    mkdir $DATA_DIR
fi

cd $DATA_DIR

wget https://thor.robots.ox.ac.uk/~vgg/data/clevrtex/clevrtex_packaged/clevrtex_full_part1.tar.gz
wget https://thor.robots.ox.ac.uk/~vgg/data/clevrtex/clevrtex_packaged/clevrtex_full_part2.tar.gz
wget https://thor.robots.ox.ac.uk/~vgg/data/clevrtex/clevrtex_packaged/clevrtex_full_part3.tar.gz
wget https://thor.robots.ox.ac.uk/~vgg/data/clevrtex/clevrtex_packaged/clevrtex_full_part4.tar.gz
wget https://thor.robots.ox.ac.uk/~vgg/data/clevrtex/clevrtex_packaged/clevrtex_full_part5.tar.gz
echo "ClevrTex downloaded to $DATA_DIR"


echo "Unzipping ClevrTex to $DATA_DIR/ClevrTex"
rm -rf ClevrTex
for file in *.tar.gz; do tar -zxf "$file"; done
for file in *.tar.gz; do rm "$file"; done
