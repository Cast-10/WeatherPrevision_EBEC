import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import pandas as pd
import utils
import trainLevel1 as t1

weather = pd.read_csv("metherology_dataset.csv")

weather = utils.setUp(weather)

model, X_val, X_test, y_val, y_test = t1.trainLevel1(weather)