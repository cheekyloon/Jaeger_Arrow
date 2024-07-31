#!/bin/bash

# Script to rename images
# using the date and time 
# from their metadata 

# Directory containing the images
image_dir="/Volumes/KH-ISW/Jaeger_Arrow/Fieldwork/20240722/CamDo/20240727-20240729/102GOPRO"
image_dir_new="/Volumes/KH-ISW/Jaeger_Arrow/Fieldwork/20240722/CamDo"

# Loop through all images in the directory
for image in "$image_dir"/*; do
    # Extract the DateTimeOriginal metadata (change this tag if necessary)
    datetime=$(exiftool -DateTimeOriginal -s3 "$image")
    
    # Check if the metadata was found
    if [ -z "$datetime" ]; then
        echo "No DateTimeOriginal found for $image. Skipping."
        continue
    fi

    # Replace colons and spaces to make a valid filename
    formatted_datetime=$(echo $datetime | sed 's/://g' | sed 's/ /_/g')

    # Get the file extension
    extension="${image##*.}"

    # Construct the new filename
    new_filename="${formatted_datetime}.${extension}"

    # Rename the image
    mv "$image" "$image_dir_new/$new_filename"

    echo "Renamed $image to $new_filename"
done
