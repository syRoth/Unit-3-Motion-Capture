# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:54:09 2019

@author: Samuel Rothstein

This code will record an image of the joint of interest from a webcam
"""
#import need libraries
import cv2
import numpy
import os.path

#capture an image from the camera; 0 = built in webcam or external camera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

#create window to display image
cv2.namedWindow('frame')
cv2.namedWindow('mask')
cv2.namedWindow('filter')
cv2.namedWindow('min')
cv2.namedWindow('max')

#Min
cv2.createTrackbar('minR','min', 0, 255, lambda x:None)
cv2.createTrackbar('minG','min', 0, 255, lambda x:None)
cv2.createTrackbar('minB','min', 0, 255, lambda x:None)
#Max
cv2.createTrackbar('maxR','max', 255, 255, lambda x:None)
cv2.createTrackbar('maxG','max', 255, 255, lambda x:None)
cv2.createTrackbar('maxB','max', 255, 255, lambda x:None)

keypressed = 1
while keypressed != 27:
    #read a frame from the video capture
    ret, frame = cap.read()
    
    #create lower and upper bounds using trackbar position
    lower = numpy.array([cv2.getTrackbarPos('minB','min'), cv2.getTrackbarPos('minG','min'), cv2.getTrackbarPos('minR','min')])
    upper = numpy.array([cv2.getTrackbarPos('maxB','max'), cv2.getTrackbarPos('maxG','max'), cv2.getTrackbarPos('maxR','max')])
    
    #create mask and filter
    mask = cv2.inRange(frame, lower, upper)
    filtered = cv2.bitwise_and(frame,frame, mask=mask)
    
    #show frame, mask, and filtered image
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('filter',filtered)
    
    #wait for button press
    keypressed = cv2.waitKey(1)
    
if keypressed == 27:
    #destroy windows
    cv2.destroyAllWindows()
    cap.release()