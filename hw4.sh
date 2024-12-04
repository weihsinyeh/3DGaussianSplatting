#!/bin/bash

# TODO - run your inference Python3 code
# bash hw4.sh $1 $2
# $1: path to the folder of split (e.g., */dataset/private_test)
# It also contains the folder of sparse/0/. The camera poses are in sparse/0/cameras.txt and sparse/0/images.txt. You should predict novel views base on the private test split.
# $2: path of the folder to put output images (e.g., xxxxxxxxx.png, please follows sparse/0/images.txt to name your output images.)
# The filename should be {id}.png (e.g. xxxxxxxxx.png). The image size should be the same as training set.

# bash hw4.sh /project/g/r13922043/hw4_dataset/dataset/public_test ./final_test
# bash hw4.sh /project/g/r13922043/hw4_dataset/dataset/private_test ./final_private_test
# python gaussian-splatting/metrics.py --gt_dir /project/g/r13922043/hw4_dataset/dataset/public_test/images/ --renders_dir ./final_test/
# python gaussian-splatting/evaluation.py --source_path /project/g/r13922043/hw4_dataset/dataset/public_test --output_dir ./output_dir
# python grade.py ./final_test /project/g/r13922043/hw4_dataset/dataset/public_test/images/
python3 gaussian-splatting/evaluation.py --source_path $1 --output_dir $2