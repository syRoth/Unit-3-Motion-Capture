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

"""
Filter Settings for different colors in form of min[H,S,V] max[H,S,V]:
Green: [77,32,0] [101,202,255]
Yellow: [20,73,75] [303,255,255]
Blue: [86,96,78] [117,227,255]
Pink: [3,137,59] [11,255,255]

First try Buttons:
Blue: [100,25,10] [180,255,225]
Red: [0,175,50] [15,255,215]
Green: [50,65,60] [140,230,120]
Yellow: [10,150,20] [30,255,255]

Second try Buttons:
Blue: [106,0,0] [126,0,0]
Red: [3,155,0] [15,255,255]
Green: [70,60,80] [110,170,255]
Yellow: [20,70,0] [35,255,255]
"""

class Color():
    def __init__(self, color_min, color_max):
        self.min = color_min
        self.max = color_max
        
    def color_filter(self, hsv, frame):
        mask = cv2.inRange(hsv, self.min, self.max)
        self.filter = cv2.bitwise_and(frame,frame, mask=mask)
        return(self.filter)

    def centroid(self, hsv, frame):
        # Find centroid of dilated image
        image = self.color_filter(hsv, frame)
        kernel = numpy.ones((5,5), numpy.uint8)
        eroded = cv2.erode(image, kernel, iterations=3)
        dilated = cv2.dilate(eroded, kernel, iterations=7)
        #eroded = cv2.erode(dilated, kernel, iterations=5)
        #dilated = cv2.dilate(eroded, kernel, iterations=7)
        image = dilated
        
        # Make it grayscale, threshold it, find the moments
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray_image,0,255,0)
        cv2.imshow('test',thresh)
        M = cv2.moments(thresh)
        print(M["m00"])
        # So we don't divide by zero
        if(M["m00"] != 0):
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print("X:" + str(cX) + "  Y:" + str(cY))
        else:
            if ('cX' in globals()) and ('cY' in globals()):
                pass
            else:
                cX, cY = 0, 0
        # Make the circle in the coordinates
        cv2.circle(image, (cX, cY), 5, (255,255,255), -1)
        return(image)

def run_filter():
    # Capture an image from the camera; 0 = built in webcam or external camera
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    
    # Create windows for display
    cv2.namedWindow('frame')
    cv2.namedWindow('filter')
    cv2.namedWindow('erode/dilate')
    cv2.namedWindow('centroid')
    
    keypressed = 1
    
    # Initialize colors
    color0 = Color(numpy.array([106,0,0]), numpy.array([126,255,255]))
    color1 = Color(numpy.array([3,155,0]), numpy.array([15,255,255]))
    color2 = Color(numpy.array([70,60,80]), numpy.array([110,170,255]))
    color3 = Color(numpy.array([20,70,0]), numpy.array([35,225,255]))
    
    while keypressed != 27:
        # Read a frame from the video capture
        ret, frame = cap.read()
        
        # Create HSV converted frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create each color filter
        color0_filter = color0.color_filter(hsv, frame)
        color1_filter = color1.color_filter(hsv, frame)
        color2_filter = color2.color_filter(hsv, frame)
        color3_filter = color3.color_filter(hsv, frame)
        
        # Combine all color filters
        filtered = cv2.bitwise_or(cv2.bitwise_or(cv2.bitwise_or(color0_filter, color1_filter), color2_filter), color3_filter)
        
        # Erode/Dilate
        kernel = numpy.ones((5,5), numpy.uint8)
        eroded = cv2.erode(filtered, kernel, iterations=3)
        dilated = cv2.dilate(eroded, kernel, iterations=5)
        
        # Find centroids
        centroid = color3.centroid(hsv, frame)
        
        # Show images
        cv2.imshow('frame', frame)
        cv2.imshow('filter', filtered)
        cv2.imshow('erode/dilate', dilated)
        cv2.imshow('centroid', centroid)
        
        # Wait for button press
        keypressed = cv2.waitKey(1)
        
    if keypressed == 27:
        # Destroy windows
        cv2.destroyAllWindows()
        cap.release()

def find_filter():
    # Capture an image from the camera; 0 = built in webcam or external camera
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    
    # Create windows for display
    cv2.namedWindow('HSV_filter')
    cv2.namedWindow('min')
    cv2.namedWindow('max')

    # Min HSV trackbars
    cv2.createTrackbar('minH','min', 0, 180, lambda x:None)
    cv2.createTrackbar('minS','min', 0, 255, lambda x:None)
    cv2.createTrackbar('minV','min', 0, 255, lambda x:None)
    # Max HSV trackbars
    cv2.createTrackbar('maxH','max', 180, 180, lambda x:None)
    cv2.createTrackbar('maxS','max', 255, 255, lambda x:None)
    cv2.createTrackbar('maxV','max', 255, 255, lambda x:None)
    
    keypressed = 1
    while keypressed != 27:
        # Read a frame from the video capture
        ret, frame = cap.read()
        
        # Create HSV converted frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Update lower and upper HSV values from trackbar
        lowerHSV = numpy.array([cv2.getTrackbarPos('minH','min'),cv2.getTrackbarPos('minS','min'),cv2.getTrackbarPos('minV','min')])
        upperHSV = numpy.array([cv2.getTrackbarPos('maxH','max'),cv2.getTrackbarPos('maxS','max'),cv2.getTrackbarPos('maxV','max')])
        
        # Create HSV mask and filter
        HSV_mask = cv2.inRange(hsv, lowerHSV, upperHSV)
        HSV_filter = cv2.bitwise_and(frame,frame, mask=HSV_mask)
        
        # Show image
        cv2.imshow('HSV_filter', HSV_filter)
        
        # Wait for button press
        keypressed = cv2.waitKey(1)
        
    if keypressed == 27:
        # Destroy windows
        cv2.destroyAllWindows()
        cap.release()


run_filter()