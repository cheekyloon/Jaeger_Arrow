#!/bin/bash

# Script to read metadata from images

# Directory containing the images
image_dir="/Volumes/KH-ISW/Jaeger_Arrow/Fieldwork/20240722/CamDo/20240723-20240724/101GOPRO"

# Loop through all images in the directory
for image in "$image_dir"/*; do
    echo "Metadata for $image:"
    identify -verbose "$image"
    echo ""
done



