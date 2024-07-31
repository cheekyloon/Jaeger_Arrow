#!/usr/bin/env python 

# Make a movie from a serie of images
# from the CamDo, with the name of each 
# image on each frame 
# Image names contain date and time
# of the image creation

#=========================================
### import modules
import cv2
import os

#=========================================
# name of movie file
mfile     = '20240723-20240729'
### path of movie 
mname  = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/Instruments/CamDo/movie/' + mfile + '.avi'
### path to directory
dirF      = '/Volumes/KH-ISW/Jaeger_Arrow/Fieldwork/20240722/CamDo/'
#dirF      = '/Users/sandy/Desktop/test/'
### number of frame per second
fps       = 5
### define text properties
# coordinates of the bottom-left corner of the text string in the image
org       = (2200, 4700)
# font
font      = cv2.FONT_HERSHEY_SIMPLEX
# font scale factor that is multiplied by the font-specific base size
fontScale = 10
# color of text string to be drawn
fontColor = (255, 255, 255)  # White color
# Line thickness of 2 px 
thickness = 20
# type of the line to be used
lineType = 2

#=========================================
### Get the list of image files
images = sorted([img for img in os.listdir(dirF) if img.endswith(".JPG")])
### Sort images by name if they are numbered sequentially
images.sort()
### Set the frame rate and size (width, height)
frame = cv2.imread(os.path.join(dirF, images[0]))
height, width, layers = frame.shape
### start movie writer
# CODEC MJPG limit the number of images in the movie
# I replaced it with the CODEC XVID
# The movie created with XVID are not read
# with QuickTime Player, only with VLC
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(mname, fourcc, fps, (width,height))
### create the movie
for image in images:
    img = cv2.imread(os.path.join(dirF, image))
    text = os.path.splitext(image)[0].replace('_', ', ')
    cv2.putText(img, text,
            org,
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
            )
    video.write(img)
### Release the video writer object
video.release()
### 
cv2.destroyAllWindows()

print("Video generated successfully!")

