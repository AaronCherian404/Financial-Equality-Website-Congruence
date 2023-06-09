import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import f1_score
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
print(loaded_model.predict([[10000.0,0,0.65,169.87,4.0,12.0,1,2400000.0,2,2]]))