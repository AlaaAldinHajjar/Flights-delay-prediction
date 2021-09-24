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
import matplotlib.pyplot as plt


def f_e(df):
    # Converting the dates
    df['Scheduled depature time'] = pd.to_datetime(
        df['Scheduled depature time'])
    df['Scheduled arrival time'] = pd.to_datetime(df['Scheduled arrival time'])
    df['Scheduled depature time'] = pd.to_datetime(
        df['Scheduled depature time'], format='%Y-%m-%d%H:%M:%S', errors='coerce')
    df['Scheduled arrival time'] = pd.to_datetime(
        df['Scheduled arrival time'], format='%Y-%m-%d%H:%M:%S', errors='coerce')
    de = df['Scheduled arrival time']-df['Scheduled depature time']
    de = de.dt.total_seconds()/3600
    df['flight duration'] = de
    df['Delay'] = df['Delay']/60
    df['Scheduled depature time_year'] = df['Scheduled depature time'].dt.year
    df['Scheduled depature time_month'] = df['Scheduled depature time'].dt.month
    df['Scheduled depature time_week'] = df['Scheduled depature time'].dt.week
    df['Scheduled depature time_day'] = df['Scheduled depature time'].dt.day
    df['Scheduled depature time_houre'] = df['Scheduled depature time'].dt.hour
    df['Scheduled depature time_min'] = df['Scheduled depature time'].dt.minute
    df['Scheduled depature time_dow'] = df['Scheduled depature time'].dt.dayofweek
    df['Scheduled arrival time_year'] = df['Scheduled arrival time'].dt.year
    df['Scheduled arrival time_month'] = df['Scheduled arrival time'].dt.month
    df['Scheduled arrival time_week'] = df['Scheduled arrival time'].dt.week
    df['Scheduled arrival time_day'] = df['Scheduled arrival time'].dt.day
    df['Scheduled arrival time_houre'] = df['Scheduled arrival time'].dt.hour
    df['Scheduled arrival time_min'] = df['Scheduled arrival time'].dt.minute
    df['Scheduled arrival time_dow'] = df['Scheduled arrival time'].dt.dayofweek
    df = df.drop(['Scheduled arrival time', 'Scheduled depature time'], axis=1)
    print(df.head(10))
    return df
