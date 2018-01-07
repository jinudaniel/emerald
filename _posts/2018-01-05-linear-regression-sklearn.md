---
title: Linear Regression using Scikit-learn
comments: true
---

>Machine Learning: Field of study that gives computers the ability to learn without being explicitly programmed.
> -- <cite>Arthur Samuel (1959)</cite>

Machine learning is an application of Artificial Intelligence (AI). The focus of machine learning is to train algorithms to learn patterns and make predictions from data. Machine learning is especially valuable because it lets us use computers to automate decision-making processes.  
Netflix and Amazon use machine learning to make new product recommendations. Banks use machine learning to detect fraudulent activity in credit card. Healthcare industry is making use of machine learning to detect diseases and to montor and assess patients.

### Linear Regression
Linear Regression is a supervised learning problem where the answer to be learned is a continuous value. Linear regression is used to predict the value of an outcome variable Y based on one or more input predictor variables X.  The aim is to establish a linear relationship (a mathematical formula) between the predictor variable(s) and the response variable, so that, we can use this formula to estimate the value of the response Y, when only the predictors (Xs) values are known.

In this tutorial, we will implement a simple linear regression algorithm in Python using [Scikit-learn](http://scikit-learn.org/stable/){:target="_blank"}, a machine learning tool for Python.

#### Loading the dataset.
The dataset that we will use will help us in determining if there is a relation between Brain weight(grams) and Head size(cubic cm). The data set is associated with the following paper: *A Study of the Relations of the Brain to to the Size of the Head*, by R.J. Gladstone, published in Biometrika, 1905.  It's a rather quaint data set, created well over a century ago and can be downloaded from [here](https://github.com/jinudaniel/machine-learning-examples/blob/master/dataset_brain.txt){:target="_blank"}.
{% highlight python linenos %}
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
{% endhighlight %}

{% highlight python linenos %}
df = pd.read_csv('dataset_brain.txt', comment='#', sep='\s+')
df.head()
{% endhighlight %}

<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>age-group</th>
      <th>head-size</th>
      <th>brain-weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>4512</td>
      <td>1530</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>3738</td>
      <td>1297</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>4261</td>
      <td>1335</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>3777</td>
      <td>1282</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>4177</td>
      <td>1590</td>
    </tr>
  </tbody>
</table>
</div>
  
#### Visualizing data
{% highlight python linenos %}
plt.scatter(df['head-size'], df['brain-weight'])
plt.xlabel('Head size (cm^3)')
plt.ylabel('Brain weight (grams)');
{% endhighlight %}
![Scatter Plot](/img/lr_sklearn_1.png "Scatter Plot")

#### Preparing the data
{% highlight python linenos %}
y = df['brain-weight'].values
y.shape
{% endhighlight %}
(237,)

{% highlight python linenos %}
X = df['head-size'].values
X = X.reshape(X.shape[0], 1)
X.shape
{% endhighlight %}
(237, 1)

A general practice is to split your data into a training and test set. You train/tune your model with your training set and test how well it generalizes to data it has never seen before with your test set. We will make use of scikit learn's `train_test_split` method to achieve this.
{% highlight python linenos %}
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=100)
{% endhighlight %}

Let us visualize the training and test set data.
{% highlight python linenos %}
plt.scatter(X_train, y_train, c='blue', marker='o')
plt.scatter(X_test, y_test, c='red', marker='p')
plt.xlabel('Head size (cm^3)')
plt.ylabel('Brain weight (grams)');
{% endhighlight %}
![Train and Test Data](/img/lr_sklearn_2.png "Train and Test Data")

#### Training the Model
{% highlight python linenos %}
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
{% endhighlight %}

#### Evaluating the model
$$R^{2}$$ is a statistical measure of how close the data are to the fitted regression line. It is also known as the coefficient of determination.  
In general, the higher the R-squared, the better the model fits your data. Scikit learn's `score` function returns the coefficient of determination $$R^{2}$$ of the prediction.

{% highlight python linenos %}
lr.score(X_test, y_test)
{% endhighlight %}
0.68879769950865688

Let us also plot the regression line which will help us in giving a better intuition about Linear Regression.
{% highlight python linenos %}
min_pred = X_train.min() * lr.coef_ + lr.intercept_
max_pred = X_train.max() * lr.coef_ + lr.intercept_

plt.scatter(X_train, y_train, c='blue', marker='o')
plt.plot([X_train.min(), X_train.max()],
         [min_pred, max_pred],
         color='red',
         linewidth=4)
plt.xlabel('Head size (cm^3)')
plt.ylabel('Brain weight (grams)');
{% endhighlight %}
![Linear Regression](/img/lr_sklearn_3.png "Linear Regression")

Entire code is available on [github](https://github.com/jinudaniel/machine-learning-examples/blob/master/linear_regression_sklearn.ipynb){:target="_blank"}.  

To know more about Linear Regression check out this [link](https://jakevdp.github.io/PythonDataScienceHandbook/05.06-linear-regression.html).