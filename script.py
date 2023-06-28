from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

import pandas as pd
import numpy as np

import csv


def bulk_regression():
    tickers_df = pd.read_csv("TICKERS.csv")

    with open("output.csv", mode="w", newline="") as file:
        writer = csv.writer(file)

        # write header
        writer.writerow(["ticker", "value", "pred"])

        for index, row in tickers_df.iterrows():
            train_set_length = 360
            test_set_length = 40
            days_after_training = 30

            dataset_path = f'data/{row["ticker"]}.SAO.csv'
            df = pd.read_csv(dataset_path)

            y = df.iloc[
                -(test_set_length + train_set_length) : -test_set_length, 3:4
            ].values
            X = np.array([[i for i in reversed(range(len(y)))]]).reshape(-1, 1)

            fy = df.iloc[
                -(test_set_length - days_after_training)
                - 1 : -(test_set_length - days_after_training),
                3:4,
            ].values
            
            fX = len(X) - 1 + days_after_training

            poly_lin_reg = train_regression(X, y)
            y_pred = poly_lin_reg.predict(get_poly_features(X, y, features=X))

            fX_poly = get_poly_features(
                X, y, features=[[fX]]
            )

            fy_pred = poly_lin_reg.predict(fX_poly)
            
            auxX = [[i for i in range(len(X)-1, fX) ]]
            poly_auxX = get_poly_features(X, y, features=auxX)
            auxY_pred = poly_lin_reg.predict(poly_auxX)

            if index == 0:
                plt.figure(figsize=(10, 5))

                plt.scatter(X, y, color="red", s=6)
                plt.plot(X, y_pred, color="blue")
                plt.scatter(fX, fy, color="green")
                plt.scatter(fX, fy_pred, color="black")
                plt.title("Polynomial Regression")
                plt.xlabel("date")
                plt.ylabel("price")
                plt.savefig("chart.jpg", format="jpg")

                break
            
            # writer.writerow([row["ticker"], fy[0][0], round(fy_pred[0][0], 4)])


def get_poly_features(X, y, features):
    degree = get_best_degree(X, y)

    poly_reg = PolynomialFeatures(degree)

    return poly_reg.fit_transform(features)


def train_regression(X, y):
    X_poly = get_poly_features(X, y, features=X)

    poly_lin_reg = LinearRegression()
    poly_lin_reg.fit(X_poly, y)

    return poly_lin_reg


def get_best_degree(X, y):
    r2A_hist = []
    r2_hist = []
    degrees = []

    n = len(X)
    k = 1

    for num in range(2, 15):
        hist_poly_reg = PolynomialFeatures(degree=num)

        hist_X_poly = hist_poly_reg.fit_transform(X)

        hist_poly_lin_reg = LinearRegression()
        hist_poly_lin_reg.fit(hist_X_poly, y)

        r2 = r2_score(y, hist_poly_lin_reg.predict(hist_X_poly))
        adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))

        r2_hist.append(r2)
        r2A_hist.append(adj_r2)
        degrees.append(num)

    return degrees[r2A_hist.index(max(r2A_hist))]


def main():
    bulk_regression()


main()
