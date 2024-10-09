# Estimating Service Metrics from Device Measurements

## Project Overview

This project investigates the service quality of a *Video-on-Demand *(VoD) service using device statistics from the server and a service metric from the client side. 

The goal is to estimate the video frame rate on the client side using **machine-learning techniques**. We approach this problem as a regression task, aiming to map device statistics to video frame rate. 

The project is divided into five tasks, which cover various data science techniques such as data exploration, linear regression, feature selection, dimensionality reduction, and neural network regression.

## Tasks Overview

### Task 1: Data Exploration

Exploration the dataset consisting of **3600 observations** of **12 different device statistics** and the video frame rate as the target variable.

We computed basic statistics (mean, standard deviation, percentiles) for each feature and produced visualizations such as time series plots, box plots, density plots, and histograms.

### Task 2: Estimating Service Metrics Using Linear Regression

We trained a **linear regression model ** to estimate the video frame rate from the device statistics. The dataset was split into training and test sets, and we computed the model's coefficients and the **Normalized Mean Absolute Error** (NMAE) on the test set.

We compared the linear regression model's accuracy with a naïve baseline that predicted the average frame rate. 

Additionally, we studied the relationship between the size of the training set and the estimation error, demonstrating that larger training sets reduce the error.

### Task 3: Feature Selection

In this task, we focused on reducing the number of features needed for accurate predictions. Two methods were used:

  **Optimal Method**: Exhaustive search over all subsets of features to find the subset that minimizes the estimation error.

 **Heuristic Method**: Selecting features based on their Pearson correlation with the target variable. We evaluated the error for models using different numbers of top features.

### Task 4: Dimensionality Reduction Using Principal Component Analysis (PCA)

We applied **PCA** to reduce the dimensionality of the feature space and evaluated the impact on prediction accuracy. Models were trained on reduced sets of principal components, and we compared PCA's performance to the feature selection methods from Task 3.

Although PCA required more components to achieve low error, it provided a computationally efficient alternative for dimensionality reduction.

### Task 5: Estimating Service Metrics Using Neural Networks

We implemented a **neural network regressor** to estimate the video frame rate, experimenting with different architectures and hyperparameters. 

We used hyperparameter search strategies such as *Random Search*, *Bayesian Optimization*, and *Hyperband* to find the best model configuration. The final tuned neural network outperformed the linear regression model in terms of accuracy.

## Code Structure

The code for each task is organized in separate Jupyter notebooks:

**task_I_II_1.ipynb**: Data loading, exploration, and visualization.

**task_II_III_1.ipynb**: Linear regression model training and evaluation.

**task_III_2_IV.ipynb**: Feature selection using optimal and heuristic methods. Dimensionality reduction using PCA.

**task_V.ipynb**: Neural network implementation and hyperparameter tuning.

## Results

You can find the results of this projet in the file *project_final_report.pdf*.

## Requirements

Python 3.x

Libraries: pandas, numpy, matplotlib, scikit-learn, keras, keras-tuner
