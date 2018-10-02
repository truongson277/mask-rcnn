#!/bin/bash
file="test.txt"
while IFS= read -r line
do
        # display $line or do somthing with $line
        python3 identity.py splash --weights=../../mask_rcnn_identity_0001.h5 --image="$line"
done <"$file"
