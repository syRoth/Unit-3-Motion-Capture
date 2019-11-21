# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:54:09 2019

@author: Samuel Rothstein

This code will record an image of the joint of interest from a webcam
"""
#import need libraries
import cv2

#capture an image from the camera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

#create window to display image
cv2.namedWindow('cap')


keypressed = 1
while keypressed != 27:
    #read a frame from the video capture
    ret, frame = cap.read()
    
    cv2.imshow('cap',frame)
    
    #wait for button press
    keypressed = cv2.waitKey(1)
    
if keypressed == 27:
    #destroy windows
    cv2.destroyAllWindows()
    cap.release()