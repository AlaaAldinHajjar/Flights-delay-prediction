import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def removeOutliers(datafframe, feature):

    upper_limit = datafframe[feature].mean() + 3 * datafframe[feature].std()
    lower_limit = datafframe[feature].mean() - 3 * datafframe[feature].std()
    new_train_data = datafframe[(datafframe[feature] < upper_limit) & (
        datafframe[feature] > lower_limit)]

    return new_train_data
