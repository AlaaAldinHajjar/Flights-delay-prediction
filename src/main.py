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
from Features_engineering import f_e
from Encoding import encode
from Imputer import imp
from Outliers_rem import removeOutliers

if __name__ == "__main__":
    df = pd.read_csv('flight_delay.csv')
    print(df.head(10))
    print(df.dtypes)

    # Features_engineering
    print('_________________Features_engineering_________________')
    df = f_e(df)
    print('_________________encode_________________')
    df = encode(df)
    print('_________________imputer_________________')
    df = imp(df)

    # visualizing
    print('_________________visualizing_________________')
    plt.figure(0)
    plt.scatter(df['flight duration'], df['Delay'], s=0.1)
    plt.title("Delay to flight duration")
    plt.xlabel("Flight Duration")
    plt.ylabel("Delay")
    plt.show()

    dim_reducer = PCA(n_components=2)
    df_reduced = dim_reducer.fit_transform(df)
    plt.figure(1)
    plt.scatter(df_reduced[:, 0], df_reduced[:, 1], s=0.1)
    plt.title("2D PCA")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.show()

    # Outlier Detection & Removal
    print('_________________Outlier Detection & Removal_________________')
    plt.figure(2)
    plt.title('Box Plot', fontsize=40)
    sns.boxplot(x=df['flight duration'])

    plt.figure(3)
    plt.title('Box Plot', fontsize=40)
    sns.boxplot(x=df['Destination Airport'])

    plt.figure(4)
    plt.title('Box Plot', fontsize=40)
    sns.boxplot(x=df['Depature Airport'])

    plt.figure(5)
    plt.title('Box Plot', fontsize=40)
    sns.boxplot(x=df['Delay'])

    df = removeOutliers(df, 'Delay')
    df = removeOutliers(df, 'flight duration')

    # Splitting to test and train
    print('_________________Splitting to test and train_________________')
    train = df.loc[df['Scheduled depature time_year'] < 2018]
    test = df.loc[df['Scheduled depature time_year'] == 2018]

    y_test = test['Delay']
    x_test = test[['flight duration']]
    X_test = test.drop('Delay', axis=1)
    y_train = train['Delay']
    x_train = train[['flight duration']]
    X_train = train.drop('Delay', axis=1)

    # Scale the data
    print('_________________Scale the data_________________')
    scaler = RobustScaler()
    scaler.fit_transform(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    scaler.fit_transform(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    x_train = x_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)

    # Simple Linear Regression
    print('_________________Simple Linear Regression_________________')
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    print(f"Model intercept : {regressor.intercept_}")
    print(f"Model coefficient : {regressor.coef_}")

    print('_________________test_________________')
    y_pred = regressor.predict(x_test)

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(
        metrics.mean_squared_error(y_test, y_pred)))
    print('R2-score:', r2_score(y_test, y_pred))

    print('_________________train_________________')
    y_pred = regressor.predict(x_train)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_pred))
    print('Root Mean Squared Error:', np.sqrt(
        metrics.mean_squared_error(y_train, y_pred)))
    print('R2-score:', r2_score(y_train, y_pred))

    plt.figure(6)
    plt.scatter(df['flight duration'], df['Delay'],  color='blue', s=1)
    XX = np.arange(0.0, 10.0, 0.1)
    yy = regressor.intercept_ + regressor.coef_[0]*XX
    plt.plot(XX, yy, '-r')
    plt.title("Delay to flight duration using simple Linear Regression")

    plt.xlabel("Flight Duration")
    plt.ylabel("Delay")

    # Multiple Linear Regression
    print('_________________Multiple Linear Regression_________________')
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    print(f"Model intercept : {regressor.intercept_}")
    print(f"Model coefficients : {regressor.coef_}")

    print('_________________test_________________')
    y_pred = regressor.predict(X_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(
        metrics.mean_squared_error(y_test, y_pred)))
    print('R2-score:', r2_score(y_test, y_pred))

    print('_________________train_________________')
    y_pred = regressor.predict(X_train)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_pred))
    print('Root Mean Squared Error:', np.sqrt(
        metrics.mean_squared_error(y_train, y_pred)))
    print('R2-score:', r2_score(y_train, y_pred))

    # Lasso & Ridge with regularization
    print('_________________Lasso & Ridge with regularization_________________')
    alphas = [2.2, 2, 1.5, 1.3, 1.2, 1.1, 1, 0.3, 0.1]
    losses = []
    for alpha in alphas:
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(mse)
        losses.append(mse)
    plt.figure(7)
    plt.plot(alphas, losses)
    plt.title("Lasso alpha value selection")
    plt.xlabel("alpha")
    plt.ylabel("Mean squared error")
    plt.show()

    best_alpha = alphas[np.argmin(losses)]
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    print('test')
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(
        metrics.mean_squared_error(y_test, y_pred)))
    print('R2-score:', r2_score(y_test, y_pred))

    y_pred = lasso.predict(X_train)
    print('train')
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_pred))
    print('Root Mean Squared Error:', np.sqrt(
        metrics.mean_squared_error(y_train, y_pred)))
    print('R2-score:', r2_score(y_train, y_pred))
    print("Best value of alpha:", best_alpha)

    alphas = [2.2, 2, 1.5, 1.3, 1.2, 1.1, 1, 0.3, 0.1]
    losses = []
    for alpha in alphas:
        # Write (5 lines): create a Lasso regressor with the alpha value.
        # Fit it to the training set, then get the prediction of the validation set (x_val).
        # calculate the mean sqaured error loss, then append it to the losses array
        ridge = Ridge(alpha=alpha)
        ridge.fit(x_train, y_train)
        y_pred = ridge.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        print(mse)
        losses.append(mse)
    plt.figure(8)
    plt.plot(alphas, losses)
    plt.title("Ridge alpha value selection")
    plt.xlabel("alpha")
    plt.ylabel("Mean squared error")
    plt.show()

    best_alpha = alphas[np.argmin(losses)]
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    print('test')
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(
        metrics.mean_squared_error(y_test, y_pred)))
    print('R2-score:', r2_score(y_test, y_pred))
    y_pred = ridge.predict(X_train)
    print('train')
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_pred))
    print('Root Mean Squared Error:', np.sqrt(
        metrics.mean_squared_error(y_train, y_pred)))
    print('R2-score:', r2_score(y_train, y_pred))
    print("Best value of alpha:", best_alpha)

    # Polynomial Regression with one feature (flight duration)
    print('_________________Polynomial Regression with one feature (flight duration)_________________')
    poly = PolynomialFeatures(degree=2)
    train_x_poly = poly.fit_transform(x_train)

    clf = linear_model.LinearRegression()
    train_y_ = clf.fit(train_x_poly, y_train)
    # The coefficients
    print('Coefficients: ', clf.coef_)
    print('Intercept: ', clf.intercept_)

    test_x_poly = poly.fit_transform(x_test)
    test_y_ = clf.predict(test_x_poly)
    print('_________________test_________________')
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, test_y_))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, test_y_))
    print('Root Mean Squared Error:', np.sqrt(
        metrics.mean_squared_error(y_test, test_y_)))
    print('R2-score:', r2_score(y_test, test_y_))

    test_x_poly = poly.fit_transform(x_train)
    test_y_ = clf.predict(test_x_poly)
    print('_________________train_________________')
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, test_y_))
    print('Mean Squared Error:', metrics.mean_squared_error(y_train, test_y_))
    print('Root Mean Squared Error:', np.sqrt(
        metrics.mean_squared_error(y_train, test_y_)))
    print('R2-score:', r2_score(y_train, test_y_))

    plt.figure(9)
    plt.scatter(df['flight duration'], df['Delay'],  color='blue', s=1)
    XX = np.arange(0.0, 10.0, 0.1)
    yy = clf.intercept_ + clf.coef_[1]*XX + clf.coef_[2]*np.power(XX, 2)
    plt.plot(XX, yy, '-r')
    plt.title("Delay to flight duration using Polynomial Regression")
    plt.xlabel("Flight Duration")
    plt.ylabel("Delay")

    # Polynomial Regression with all features
    print('_________________Polynomial Regression with all features_________________')
    poly_all = PolynomialFeatures(degree=2)
    train_x_poly_all = poly_all.fit_transform(X_train)

    clf_all = linear_model.LinearRegression()
    train_y_all = clf_all.fit(train_x_poly_all, y_train)
    # The coefficients
    print('Coefficients: ', clf_all.coef_)
    print('Intercept: ', clf_all.intercept_)

    test_x_poly_all = poly.fit_transform(X_test)
    test_y_all = clf_all.predict(test_x_poly_all)
    print('_________________test_________________')
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, test_y_all))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, test_y_all))
    print('Root Mean Squared Error:', np.sqrt(
        metrics.mean_squared_error(y_test, test_y_all)))
    print('R2-score:', r2_score(y_test, test_y_all))

    test_x_poly_all = poly.fit_transform(X_train)
    test_y_all = clf_all.predict(test_x_poly_all)
    print('_________________train_________________')
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, test_y_all))
    print('Mean Squared Error:', metrics.mean_squared_error(y_train, test_y_all))
    print('Root Mean Squared Error:', np.sqrt(
        metrics.mean_squared_error(y_train, test_y_all)))
    print('R2-score:', r2_score(y_train, test_y_all))

    # Using Neural Networks
    print('_________________Using Neural Networks_________________')
    model2 = Sequential()
    model2.add(Dense(16, input_dim=17, activation='relu'))
    model2.add(Dense(8, activation='relu'))
    model2.add(Dense(1, activation='relu'))
    model2.compile(loss='mean_absolute_error',
                   optimizer='adam',
                   metrics=['accuracy'])
    model2.fit(X_train, y_train, epochs=5, batch_size=32)
    predicted = model2.predict(X_test, batch_size=128)
    print('_________________test_________________')
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predicted))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, predicted))
    print('Root Mean Squared Error:', np.sqrt(
        metrics.mean_squared_error(y_test, predicted)))
    print('R2-score:', r2_score(y_test, predicted))

    predicted = model2.predict(X_train, batch_size=128)
    print('_________________train_________________')
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, predicted))
    print('Mean Squared Error:', metrics.mean_squared_error(y_train, predicted))
    print('Root Mean Squared Error:', np.sqrt(
        metrics.mean_squared_error(y_train, predicted)))
    print('R2-score:', r2_score(y_train, predicted))
