import numpy as np
from skimage import feature as skif
import cv2
import joblib
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix,roc_curve, auc

def lbp_histogram(image,P=8,R=1,method = 'nri_uniform'):
    '''
    image: shape is N*M 
    '''
    lbp = skif.local_binary_pattern(image, P,R, method) # lbp.shape is equal image.shape
    # cv2_imshow(lbp)
    max_bins = int(lbp.max() + 1) # max_bins is related P
    hist,_= np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))
    # plt.hist(y_h, bins = max_bins) 
    # plt.title("histogram") 
    # plt.show()
    return hist

def save_feature(image_path,label,path_feature,color_space):
  feature_label = []
  image = cv2.imread(image_path)

  image_ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
  y_h = lbp_histogram(image_ycbcr[:,:,0]) # y channel
  cb_h = lbp_histogram(image_ycbcr[:,:,1]) # cb channel
  cr_h = lbp_histogram(image_ycbcr[:,:,2]) # cr channel

  image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  h_h = lbp_histogram(image_hsv[:,:,0]) # h channel
  s_h = lbp_histogram(image_hsv[:,:,1]) # s channel
  v_h = lbp_histogram(image_hsv[:,:,2]) # v channel

  if color_space == 'YCBCR':
    feature = np.concatenate((y_h,cb_h,cr_h))
  elif color_space == 'HSV':
    feature = np.concatenate((h_h, s_h, v_h))
  elif color_space == 'YCBCR_HSV':
    feature = np.concatenate((y_h,cb_h,cr_h, h_h, s_h, v_h))

  feature_label.append(np.append(feature,np.array(label)))
  np.save(path_feature,np.array(feature_label))

def metric(predict_proba,labels):
  predict =  np.greater(predict_proba[:,1],0.5)
  tn, fp, fn, tp = confusion_matrix(labels,predict).flatten()
  acc = (tp+tn)/(tp+tn+fp+fn)
  far = fp / (fp + tn) # apcer
  frr = fn / (tp + fn) # bpcer
  hter=(far+frr) / 2 # acer

  fpr, tpr, threshold = roc_curve(labels, predict_proba[:,1])
  auc_v = auc(fpr, tpr) # area under curve
  dist = abs((1-fpr) - tpr)
  eer = fpr[np.argmin(dist)]

  return acc,eer,hter

def load_feature_label(file_name):
  feature_label = np.load(file_name)
  return feature_label[:,:-1],feature_label[:,-1].astype(np.uint8)

def test_one(file_name, color_space):
  test_feature,test_label = load_feature_label(file_name)

  if color_space == 'YCBCR':
    model = joblib.load("./model_ycbcr.m")
  elif color_space == 'HSV':
    model = joblib.load("./model_hsv.m")
  elif color_space == 'YCBCR_HSV':
    model = joblib.load("./model_ycbcr_hsv.m")

  predict_proba = model.predict_proba(test_feature)
  predict = model.predict(test_feature)
  return predict_proba, predict