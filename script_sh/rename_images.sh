#!/bin/bash

# Script to rename images
# using the date and time 
# from their metadata 

# Directory where to move renamed images
dir_img_new="/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/Instruments/CamDo/images/20231101"

# Make a loop over the directory containing the raw images
for dir_img in /Users/sandy/Documents/ISW_projects/Jaeger_Arrow/Instruments/CamDo/images/20231101/*; do

    # Loop through all images in each directory
    for image in "$dir_img"/*; do

        # Extract the DateTimeOriginal metadata 
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
        mv "$image" "$dir_img_new/$new_filename"

        echo "Renamed $image to $new_filename"
    done

done
