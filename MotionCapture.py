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

color0 = numpy.array([0,0,0])
color1 = numpy.array([0,0,0])
color2 = numpy.array([0,0,0])
color3 = numpy.array([0,0,0])

#create windows to display image (filtered and original)
cv2.namedWindow('frame')
cv2.namedWindow('rgbFilter')
cv2.namedWindow('hsvFilter')
cv2.namedWindow('min')
cv2.namedWindow('max')

#Min RGB
cv2.createTrackbar('minR','min', 0, 255, lambda x:None)
cv2.createTrackbar('minG','min', 0, 255, lambda x:None)
cv2.createTrackbar('minB','min', 0, 255, lambda x:None)
#Max RGB
cv2.createTrackbar('maxR','max', 255, 255, lambda x:None)
cv2.createTrackbar('maxG','max', 255, 255, lambda x:None)
cv2.createTrackbar('maxB','max', 255, 255, lambda x:None)

#Min HSV
cv2.createTrackbar('minH','min', 0, 180, lambda x:None)
cv2.createTrackbar('minS','min', 0, 255, lambda x:None)
cv2.createTrackbar('minV','min', 0, 255, lambda x:None)
#Max HSV
cv2.createTrackbar('maxH','max', 180, 180, lambda x:None)
cv2.createTrackbar('maxS','max', 255, 255, lambda x:None)
cv2.createTrackbar('maxV','max', 255, 255, lambda x:None)

keypressed = 1
while keypressed != 27:
    #read a frame from the video capture
    ret, frame = cap.read()
    #Create HSV converted frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #create lower and upper bounds using trackbar position for RGB and HSV
    lowerRGB = numpy.array([cv2.getTrackbarPos('minB','min'), cv2.getTrackbarPos('minG','min'), cv2.getTrackbarPos('minR','min')])
    upperRGB = numpy.array([cv2.getTrackbarPos('maxB','max'), cv2.getTrackbarPos('maxG','max'), cv2.getTrackbarPos('maxR','max')])
    
    lowerHSV = numpy.array([cv2.getTrackbarPos('minH','min'),cv2.getTrackbarPos('minS','min'),cv2.getTrackbarPos('minV','min')])
    upperHSV = numpy.array([cv2.getTrackbarPos('maxH','max'),cv2.getTrackbarPos('maxS','max'),cv2.getTrackbarPos('maxV','max')])
        
    #create mask and filter for RGB and HSV
    maskRGB = cv2.inRange(frame, lowerRGB, upperRGB)
    filterRGB = cv2.bitwise_and(frame,frame, mask=maskRGB)
    
    maskHSV = cv2.inRange(hsv, lowerHSV, upperHSV)
    filterHSV = cv2.bitwise_and(frame,frame, mask=maskHSV)
    
    #show frame and filtered images
    cv2.imshow('frame',frame)
    
    cv2.imshow('rgbFilter',filterRGB)
    cv2.imshow('hsvFilter',filterHSV)
    
    #wait for button press
    keypressed = cv2.waitKey(1)
    
if keypressed == 27:
    #destroy windows
    cv2.destroyAllWindows()
    cap.release()