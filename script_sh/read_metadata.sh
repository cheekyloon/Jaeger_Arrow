#!/bin/bash

# Script to read metadata from images

# Directory containing the images
dir_img="/Volumes/KH-ISW/Jaeger_Arrow/Fieldwork/20240722/CamDo/20240723-20240724/101GOPRO"

# Loop through all images in the directory
for image in "$dir_img"/*; do
    echo "Metadata for $image:"
    identify -verbose "$image"
    echo ""
done



