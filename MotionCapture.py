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
import math
import pysine

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

Third try Buttons:
Blue: [105,100,5] [120,255,200]
Green: [60,50,0] [100,200,255]
Red: [5,200,50] [15,255,255]
Yellow: [20,130,80] [35,255,255]


"""

class Color():
    def __init__(self, color_min, color_max):
        self.min = color_min
        self.max = color_max
        self.centroid = (0,0)
        
    def color_filter(self, hsv, frame):
        mask = cv2.inRange(hsv, self.min, self.max)
        self.filter = cv2.bitwise_and(frame,frame, mask=mask)
        return(self.filter)

    def find_centroid(self, hsv, frame, e, d):
        # Find centroid of dilated image
        #image = self.color_filter(hsv, frame)
        #kernel = numpy.ones((5,5), numpy.uint8)
        #eroded = cv2.erode(image, kernel, iterations=3)
        #dilated = cv2.dilate(eroded, kernel, iterations=7)
        #image = dilated
        # Erode/dilate the image
        image = self.erode_dilate(hsv, frame, e, d)
        
        # Make it grayscale, threshold it, find the moments
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray_image,0,255,0)
        M = cv2.moments(thresh)
        # So we don't divide by zero
        if(M["m00"] != 0):
            self.centroid = ( int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]) )
        # Make the circle in the coordinates
        cv2.circle(image, (self.centroid[0], self.centroid[1]), 5, (255,255,255), -1)
        return(image)
        
    def erode_dilate(self, hsv, frame, e, d):
        # erodes and dilates with set iterations
        image = self.color_filter(hsv, frame)
        kernel = numpy.ones((5,5), numpy.uint8)
        eroded = cv2.erode(image, kernel, iterations=e)
        dilated = cv2.dilate(eroded, kernel, iterations=d)
        return(dilated)

def run_filter():
    # Capture an image from the camera; 0 = built in webcam or external camera
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    
    # Create windows for display
    cv2.namedWindow('frame')
    #cv2.namedWindow('filter')
    #cv2.namedWindow('erode/dilate')
    #cv2.namedWindow('centroid')
    #cv2.namedWindow('connections')
    cv2.namedWindow('angle')
    
    keypressed = 1
    
    # Initialize colors
    # 0-Blue, 1-Green, 2-Red, 3-Yellow
    color0 = Color(numpy.array([105,100,25]), numpy.array([120,255,200]))
    color1 = Color(numpy.array([60,50,30]), numpy.array([100,200,255]))
    color2 = Color(numpy.array([5,210,50]), numpy.array([15,255,255]))
    color3 = Color(numpy.array([20,130,80]), numpy.array([35,255,170]))
    
    intersection = [0,0]
    
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
        
        # Erode/dilate individual colors
        color0_erode_dilate = color0.erode_dilate(hsv, frame, 4, 10)
        color1_erode_dilate = color1.erode_dilate(hsv, frame, 2, 10)
        color2_erode_dilate = color2.erode_dilate(hsv, frame, 3, 10)
        color3_erode_dilate = color3.erode_dilate(hsv, frame, 3, 10)
        
        # Combine all
        erode_dilate = cv2.bitwise_or(cv2.bitwise_or(cv2.bitwise_or(color0_erode_dilate, color1_erode_dilate), color2_erode_dilate), color3_erode_dilate)
        
        # Each centroid image
        color0_centroid = color0.find_centroid(hsv, frame, 4, 10)
        color1_centroid = color1.find_centroid(hsv, frame, 2, 10)
        color2_centroid = color2.find_centroid(hsv, frame, 3, 10)
        color3_centroid = color3.find_centroid(hsv, frame, 3, 10)
        
        # Combine centroids into one image
        centroid = cv2.bitwise_or(cv2.bitwise_or(cv2.bitwise_or(color0_centroid, color1_centroid), color2_centroid), color3_centroid)
        
        #Draw lines between centroids
        connections = centroid
        cv2.line(connections, (color0.centroid[0], color0.centroid[1]), (color1.centroid[0], color1.centroid[1]), (255,255,255), 5)
        cv2.line(connections, (color2.centroid[0], color2.centroid[1]), (color3.centroid[0], color3.centroid[1]), (255,255,255), 5)
        
        # Calculate intersection point
        if( (color1.centroid[0] - color0.centroid[0]) != 0 and (color3.centroid[0] - color2.centroid[0]) != 0):
            color01_slope = float(color1.centroid[1] - color0.centroid[1]) / (color1.centroid[0] - color0.centroid[0])
            color23_slope = float(color3.centroid[1] - color2.centroid[1]) / (color3.centroid[0] - color2.centroid[0])
            if( (color01_slope - color23_slope) != 0):
                intersection[0] = (-color1.centroid[1] + color3.centroid[1] + color01_slope*color1.centroid[0] - color23_slope*color3.centroid[0])/(color01_slope - color23_slope)
                intersection[1] = color01_slope*(intersection[0]-color1.centroid[0]) + color1.centroid[1]
        # Calculate angles (points are color0, color2, and intersection)
        angle = connections
        if( (intersection[0] - color0.centroid[0]) != 0 and (intersection[0] - color2.centroid[0]) != 0):
            # Creates two right triangles, with each line drawn being the hyptoneuse of those triangles
            # Then uses trig rules to calculate the angle
            a_height = abs(color2.centroid[1] - intersection[1])
            a_base = abs(color2.centroid[0] - intersection[0])
            a = math.degrees(math.atan(a_height / a_base))
            b_height = abs(color0.centroid[1] - intersection[1])
            b_base = abs(color0.centroid[0] - intersection[0])
            b = math.degrees(math.atan(b_height / b_base))
            theta = 180 - a - b
            cv2.putText(angle, ("Angle: " + str(theta)), (8,467), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
        
        # Show images
        cv2.imshow('frame', frame)
        #cv2.imshow('filter', filtered)
        #cv2.imshow('erode/dilate', erode_dilate)
        #cv2.imshow('centroid', centroid)
        #cv2.imshow('connections', connections)
        cv2.imshow('angle' ,angle)
        
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

def beep():
    

def test_math():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cv2.namedWindow('frame')
    
    color0 = Color(0,0)
    color0.centroid = (50,50)
    cv2.circle(frame, color0.centroid, 5, (255,0,0), -1)
    color1 = Color(0,0)
    color1.centroid = (80,100)
    cv2.circle(frame, color1.centroid, 5, (0,255,0), -1)
    color2 = Color(0,0)
    color2.centroid = (190,110)
    cv2.circle(frame, color2.centroid, 5, (0,0,255), -1)
    color3 = Color(0,0)
    color3.centroid = (150,130)
    cv2.circle(frame, color3.centroid, 5, (0,0,0), -1)
    cv2.circle(frame, (110,150), 5, (255,255,255), -1)
    
    intersection = [0,0]
    
    if( (color1.centroid[0] - color0.centroid[0]) != 0 and (color3.centroid[0] - color2.centroid[0]) != 0):
            color01_slope = float(color1.centroid[1] - color0.centroid[1]) / (color1.centroid[0] - color0.centroid[0])
            color23_slope = float(color3.centroid[1] - color2.centroid[1]) / (color3.centroid[0] - color2.centroid[0])
            if( (color01_slope - color23_slope) != 0):
                intersection[0] = (-color1.centroid[1] + color3.centroid[1] + color01_slope*color1.centroid[0] - color23_slope*color3.centroid[0])/(color01_slope - color23_slope)
                intersection[1] = color01_slope*(intersection[0]-color1.centroid[0]) + color1.centroid[1]
    # Calculate angles (points are color0, color2, and intersection)
    if( (intersection[0] - color0.centroid[0]) != 0 and (intersection[0] - color2.centroid[0]) != 0):
        a_height = abs(color2.centroid[1] - intersection[1])
        a_base = abs(color2.centroid[0] - intersection[0])
        a = math.degrees(math.atan(a_height / a_base))
        b_height = abs(color0.centroid[1] - intersection[1])
        b_base = abs(color0.centroid[0] - intersection[0])
        b = math.degrees(math.atan(b_height / b_base))
        theta = 180 - a - b
            
    keypressed = 1
    while keypressed != 27:
        cv2.imshow('frame',frame)
        keypressed = cv2.waitKey(1)
    if keypressed == 27:
        cv2.destroyAllWindows()
        cap.release()
        
run_filter()