from keras.models import load_model
import os

#path to model
path = "/home/andreas/Desktop/bitbucket_code/binder-andreas/Model/Generator.py"

if os.path.isfile(path):
    model = load_model(path)
    model.summary()



