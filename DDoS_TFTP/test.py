import np 
import joblib

encoder = joblib.load("encoder(tftp).joblib")
y_train = joblib.load("Combined_Data/y_train.npy")
y_test = joblib.load("Combined_Data/y_train.npy")