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

"""Filter Settings for different colors in form of min[H,S,V] max[H,S,V]:
Green: [77,32,0] [101,202,255]
Yellow: [20,73,75] [303,255,255]
Blue: [86,96,78] [117,227,255]
Pink: [3,137,59] [11,255,255]
"""

#capture an image from the camera; 0 = built in webcam or external camera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

#Filter ranges for each color (HSV)
color0_min = numpy.array([77,32,0])
color0_max = numpy.array([101,202,255])
color1_min = numpy.array([20,73,75])
color1_max = numpy.array([30,255,255])
color2_min = numpy.array([86,96,78])
color2_max = numpy.array([117,227,255])
color3_min = numpy.array([3,137,59])
color3_max = numpy.array([11,255,255])

#create windows to display image (filtered and original)
cv2.namedWindow('frame')
cv2.namedWindow('filter')
"""
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
"""

keypressed = 1
while keypressed != 27:
    #read a frame from the video capture
    ret, frame = cap.read()
    #Create HSV converted frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    """
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
    """
    
    #Create masks for each color using their predeclared restrictions
    color0_mask = cv2.inRange(hsv, color0_min, color0_max)
    color1_mask = cv2.inRange(hsv, color1_min, color1_max)
    color2_mask = cv2.inRange(hsv, color2_min, color2_max)
    color3_mask = cv2.inRange(hsv, color3_min, color3_max)
    
    #Create filtered image for each individual color
    color0_filter = cv2.bitwise_and(frame,frame, mask=color0_mask)
    color1_filter = cv2.bitwise_and(frame,frame, mask=color1_mask)
    color2_filter = cv2.bitwise_and(frame,frame, mask=color2_mask)
    color3_filter = cv2.bitwise_and(frame,frame, mask=color3_mask)
    
    #Combine each filtered image to have one master filter
    filtered = cv2.bitwise_or(cv2.bitwise_or(cv2.bitwise_or(color0_filter, color1_filter), color2_filter), color3_filter)
    
    #show frame and filtered images
    cv2.imshow('frame',frame)
    cv2.imshow('filter',filtered)
    
    """
    cv2.imshow('rgbFilter',filterRGB)
    cv2.imshow('hsvFilter',filterHSV)
    """
    
    #wait for button press
    keypressed = cv2.waitKey(1)
    
if keypressed == 27:
    #destroy windows
    cv2.destroyAllWindows()
    cap.release()