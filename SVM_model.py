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
import pickle


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


#Import training data
images=pickle.load( open( "X_data.p", "rb" ) )
labels=pickle.load( open( "Y_data.p", "rb" ) )

#Set HOG paramaters
orient = 8
pix_per_cell = 8
cell_per_block = 2

#Shell lists
X_data=[]
HOG_features=[]
spatial_features=[]
color_hist=[]

#Extract features from images
for image in images:
	#Get HOG features
	HOG_features.append(get_HOG_features(image,orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
	#Get HLS features
	spatial_features.append(get_spatial_bin(image))
	#get color histograms
	color_hist.append(get_color_hist(image))

#Convert to arrays
HOG_features=np.array(HOG_features)
spatial_features=np.array(HLS_features)
color_hist=np.array(color_hist)

#Concatenate data
feature_list=[HOG_features,spatial_features,color_hist]
X = np.column_stack(feature_list).astype(np.float64)

#Normalize data
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)

#Split into train and test and shuffle 
X_train, X_test, Y_train, Y_test = train_test_split(scaled_X, labels, test_size=0.25, random_state=42)

#Build out classifier and train
clf=SVC(kernel='linear')
clf.fit(X_train,Y_train)

#predict for Acc score
pred=clf.predict(X_test)

#Acc score
acc=accuracy_score(pred,Y_test)
print(acc)

#Print report 
acc=classification_report(Y_test,pred)
print(acc)

#save via pickle
filename = 'SVM_model.sav'
pickle.dump(clf, open(filename, 'wb'))









