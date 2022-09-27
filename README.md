# Grocery Item Sales Amount Forecasting

## Introduction
This business problem is one of the data science problems from the Kaggle Competition. The competition link is https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/overview

The objective is to predict the sales amount on over 200,000 different items over hundreds of stores.

## Method
The dataset is a time series data consisting item sales amount of each item in each store. To make the predictions, we first try out time series model ARIMA, SARIMA, and Facebook Prophet model. However, using those models which only use historical sales amount data did not help us to achieve desirable results. Instead of just utilizing data on historical sales amount, we convert the data into tabular data where we try to incorporate store information such as geography location and population around the neighorhood and add item discounts tables. By including these information with additional feature engineering and multiple LightGBM algorithm based models, we are able to achieve higher predictive power model that makes good predictions on sales.

## Model Result
With our codes, we are able to score NRMSLE (Normalized Root Mean Squared Log Error)of 0.51620 on the test set from the grocery store, which gets us to ranked 14th among 1700 teams earning a silver medal.
