# Create a Customer Segmentation Report for Arvato Financial Services 

Capstone Project for Udacity's Data Scientist Nanodegree

### Overview

This project is available as a [blog post](https://bvcmartins.github.io/jekyll/update/2019/07/21/capstone.html)

The objective of this project is to predict, based on a broad list of features, if someone who is receiving a mailout campaign will become a customer of the company. It is devided in three parts:

1. Segment the data obtained for the general population of Germany in clusters and identify in which of these clusters the current customers fall in
2. Select people from these clusters who are not customers to receive a mailout campaign and train a supervised learning model to predict who will become a new customer
3.  Apply the model to a Kaggle competition

In part 1 I carried out the data segmentation using PCA for dimensionality reduction and K-Means for clustering. In part 2 I used three tree-based supervised learning techniques (Random Forests, Ada Boost, XGBoost) models for the prediction of new customers.

The best model was tested against the test set through a Kaggle competition and the scores were ranked using the AUC (Area under curve) metric. This metric was used because of the highly imabalanced distribution of the response variable. However, in the last part of this report, I will show why this metric is not the best choice for this problem and why the F1 metric would be more suited to measure model performance.

### Datasets

Arvato kindly provided us four datasets:

* Azdias: sample of the general german population categorized according to a variety of features involving personality traits, demographics and financial information (891221 entries, 366 features).

* Customers: classification of current customers according to the same features used for Azdias (191652 entries, 369 features). It also contains customer categorization and information about purchase preferences.

* Mailout_train: training set containing potential customers chosen to receive a mailout campaign (42962 entries, 367 features). It also contains information if the person became a customer (target variable).

* Mailout_test: testing set for the supervised learning model (42833 entries, 366 features).

Two support files for the interpretation of the features were also provided:

* DIAS Attributes - Values 2017: information about code levels for some of the features.

* DIAS Information Levels - Attributes 2017: high-level information about most (but not all) features.

## Methods

We used supervised learning methods from Scikit-learn, XGBoost and Pandas for the analysis. The
following learning methods were used:

- PCA
- K-means 
- AdaBoost Classifier
- Random Forest Classifier 
- XGBoost

## Files

* main.ipynb has the full solution

## Conclusions

Here is what we learned:

That was a quite interesting project, I enjoyed very much working on it. Here are the main conclusions I obtained from it:

* the new integer NaN implemented in Pandas helps a lot with the cleaning
* outlier removal was very important for result improvement
* PCA and k-means are good tools for qualitative data analysis, specially if you want to use visualization to generate insights. However, for this dataset, they did not lead to better predictions
* Using XGBoost with sample weight parameters was a better solution to class imbalance than SMOTE
* The best model obtained an AUC score of 0.79917 using the Test set from [Kaggle competition] (http://www.kaggle.com/t/21e6d45d4c574c7fa2d868f0e8c83140)
* F1 score presents higher fidelity to Confusion Matrix than AUC. It is more adequate for highly imbalanced datasets 
