import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from scipy.ndimage.measurements import label
from moviepy.editor import *


def get_HOG_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    #extract the HOG features

    #Convert to 64x64 and YCrCb
    img=cv2.resize(img,(64,64))
    img1=cv2.cvtColor(img,cv2.COLOR_RGB2YCrCb)

    #shell list for features
    list1=[]
     
    #Extract features and append to list1 for all channels   
    img=img1[:,:,0]
    features=(hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=False, feature_vector=feature_vec))
    list1.append(features)

    img=img1[:,:,1]
    features=(hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=False, feature_vector=feature_vec))
    list1.append(features)

    img=img1[:,:,2]
    features=(hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=False, feature_vector=feature_vec))
    list1.append(features)

    #name and ravel
    features=np.ravel(list1)
    
    #return features
    return features
def get_spatial_bin(img):
    #A functiion to extract the spatial bin

    #Convert to YCrCb
    img=cv2.cvtColor(img,cv2.COLOR_RGB2YCrCb)

    #resize and ravel
    features = cv2.resize(img, (32,32)).ravel() 

    #return features
    return features


def get_color_hist(image,nbins=32, bins_range=(0, 256)):
    #A function to get the color histogram, from Udacity quiz

    #Resize and change to YCrCb
    image=cv2.resize(image,(64,64))
    image=cv2.cvtColor(image,cv2.COLOR_RGB2YCrCb)

    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(image[:,:,0], bins=32, range=(0, 256))
    ghist = np.histogram(image[:,:,1], bins=32, range=(0, 256))
    bhist = np.histogram(image[:,:,2], bins=32, range=(0, 256))

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))

    #return the result 
    return hist_features

def extract_features(image):
    #extract features, combine and scale to be sent to the classifier later

    #HOG Parameters
    orient = 8
    pix_per_cell = 8
    cell_per_block = 2

    #Convert to arrays
    HOG_features=np.array(get_HOG_features(image,orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
    spatial_features=np.array(get_spatial_bin(image))
    color_hist=np.array(get_color_hist(image))

    #Concatenate data
    feature_list=[HOG_features,spatial_features,color_hist]
    X = np.hstack(feature_list).astype(np.float64)

    #Normalize data
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    #return scaled features
    return scaled_X

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
	#A tool to define a list of sliding windows from the udacity quiz

    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def search_windows(img, windows, clf):
	#Take the list of windows and search each one for a car, if it contains a car append the window to the list, from udacity quiz
	
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))    
        #4) Extract features for that window using single_img_features()
        features = extract_features(test_img)
        #5) Scale extracted features to be fed to classifier
        test_features = np.array(features)
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    #A function to draw boxes, from udacity quiz

    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    for i in range(len(bboxes)):
        point1=bboxes[i][0]
        point2=bboxes[i][1]
        cv2.rectangle(draw_img,point1,point2,color,thick)

    # return the image copy with boxes drawn
    return draw_img

def add_heat(heatmap, bbox_list):
    #A function to add heat to the heatmap, from udacity quiz

    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        x_1=box[0][0]
        x_2=box[1][0]
        y_1=box[0][1]
        y_2=box[1][1]
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    #Apply the threshold to the heatmap

    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    #Draw boxes from a labels function, from udacity quiz

    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def breakout_bboxes(labels):
    #a function to take a list of boxes from a labels function and return a list of just the boxes to be sent to the running average function

    dims=[]
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        img=(bbox[0],bbox[1])


        dims.append(img)
        
    # Return the image
    return dims

#Shell lists for running average
average_1=[]
average_2=[]
test_avg=[]

def loop(get_frame,t):
    #This is the pipeline, all functuons are called up as needed to produce the result image

    #call up global varibles for running average
    global average_1
    global average_2
	
    #Call up the classifier
    filename = 'SVM_model.sav'
    clf = pickle.load(open(filename, 'rb'))

	#Get the current frame
    image = get_frame(t)

	#Make a blank for the heat map
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

	#Make a list of boxes to get classified as car or not car
    window_list_1=slide_window(image,x_start_stop=[600, 1280], y_start_stop=[380, 515],xy_window=(340, 135), xy_overlap=(0.5, 0.5))
    window_list_2=slide_window(image,x_start_stop=[720, 1280], y_start_stop=[380, 580],xy_window=(85, 68), xy_overlap=(0.5, 0.5))
    window_list_3=slide_window(image,x_start_stop=[500, None], y_start_stop=[400, 610],xy_window=(80, 80), xy_overlap=(0.5, 0.5))
    window_list_4=slide_window(image,x_start_stop=[600, None], y_start_stop=[400, 610],xy_window=(50, 50), xy_overlap=(0.5, 0.5))
    window_list=window_list_1+window_list_2

    #see which windows are hot
    on_windows=search_windows(image,window_list,clf)

    #Use a heat map to help reduce noise
    heat_map=add_heat(heat,on_windows)

    #threshold that heatmap
    heat_map=apply_threshold(heat_map,1)
    
	#prep the heat_map by applying the label function
    labels=label(heat_map)

    #create a list of points to be thrown to the global average
    boxes=breakout_bboxes(labels)

    #test for multible cars and add new entries to the global varible as well as elimiate old ones as needed
    if len(boxes)==1:
        if len(average_1)<15:
            average_1.append(boxes[0])
        else:
            average_1.append(boxes[0])
            average_1.remove(average_1[0])
    elif len(boxes)==2:
        if len(average_1)<15:
            average_1.append(boxes[0])
        else:
            average_1.append(boxes[0])
            average_1.remove(average_1[0])

        if len(average_2)<15:
            average_2.append(boxes[1])
        else:
            average_2.append(boxes[1])
            average_2.remove(average_2[0])

    #Test if average len is 0 to prevet an error
    if len(average_1)!=0:
    #For the first in boxes create a list of each point to be averaged later
    #first point x
        first_point_x=[]
        for i in range(len(average_1)):
            first_point_x.append(average_1[i][0][0])
        #second point x
        second_point_x=[]
        for i in range(len(average_1)):
            second_point_x.append(average_1[i][1][0])
        #first point y
        first_point_y=[]
        for i in range(len(average_1)):
            first_point_y.append(average_1[i][0][1])
        #second point y
        second_point_y=[]
        for i in range(len(average_1)):
            second_point_y.append(average_1[i][1][1])

        #average the point lists and convert to intergers
        first_point_1=int(np.average(first_point_x)),int(np.average(first_point_y))
        second_point_1=int(np.average(second_point_x)),int(np.average(second_point_y))

        #Format 
        points=[(first_point_1,second_point_1)]

        #draw the image
        box_image=draw_boxes(image,points)
    else:
        #draw the image if average len is 0
        box_image=image

    if len(average_2)!=0:
    #For the first in boxes create a list of each point to be averaged later
    #first point x
        first_point_x=[]
        for i in range(len(average_2)):
            first_point_x.append(average_2[i][0][0])
        #second point x
        second_point_x=[]
        for i in range(len(average_2)):
            second_point_x.append(average_2[i][1][0])
        #first point y
        first_point_y=[]
        for i in range(len(average_2)):
            first_point_y.append(average_2[i][0][1])
        #second point y
        second_point_y=[]
        for i in range(len(average_2)):
            second_point_y.append(average_2[i][1][1])

        #average the point lists and convert to intergers
        first_point_2=int(np.average(first_point_x)),int(np.average(first_point_y))
        second_point_2=int(np.average(second_point_x)),int(np.average(second_point_y))

        #Format
        points=[(first_point_2,second_point_2)]

        #Draw the image
        box_image=draw_boxes(box_image,points)
    else:
        #draw the image if average len is 0
        box_image=image

    #return the final image
    return box_image
       
#File paths and fl function 
white_output = 'test new.mp4'
clip1 = VideoFileClip("/Users/johnmcconnell/Desktop/project_video copy.mp4",audio=False)
white_clip = clip1.fl(loop) 
white_clip.write_videofile(white_output, audio=False)








