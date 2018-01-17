---
title: Classification using TensorFlow's Estimator API
comments: true
---

### Classification
Classification is the task of approximating a mapping function (f) from input variables (X) to discrete output variables (y).  The output variables are often called labels or categories.  
For example, an email can be classified as *spam* or *not spam* or a transaction can e classified as *fraudulent* or *authorized*  
There are a number of classification models. Classification models include Logistic Regression, Decision Tree, Random Forest, Neural Networks etc.

### TensorFlow
TensorFlow is an open source software library for numerical computation using data flow graphs. More about tensorflow can be found in the [official documentation](https://www.tensorflow.org/get_started/){:target="_blank"}.  
**Estimators** is a high-level API that reduces much of the boilerplate code you previously needed to write when training a TensorFlow model.

#### Loading the dataset.
The dataset we'll be using is the [Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Census+Income){:target="_blank"}. This datset will help us in predicting whether the income exceeds $50K/yr based on census data.
<script src="https://gist.github.com/jinudaniel/285bb33861666d7d839ce2a19631159d.js"></script>

Since the task is Binary Classification, we will make use of Pandas `apply` function to convert the `income_bracket` column to a label whose value is 1 if the income is above $50k and 0 otherwise.
<script src="https://gist.github.com/jinudaniel/e4a326e5b8e84a6df41b93a1dd6762b7.js"></script>

#### Train and Test Split.
Let's make use of sklearn's train_test_split method to split the data into training and test set.
<script src="https://gist.github.com/jinudaniel/aff24c965fee1e5ab0586df4f5b534ef.js"></script>

#### Categorical Feature Columns
Feature column is an abstract concept of any raw or derived variable that can be used to predict the target label. To define a feature column for a categorical feature, we can create a `CategoricalColumn` using the tf.feature_column API.  

If we know the possible values for a Categorical feature columns then we can make use of `categorical_column_with_vocabulary_list` but if we are not aware of all the possible values then we can make use of `categorical_column_with_hash_bucket` with the hash_bucket_size defined.

<script src="https://gist.github.com/jinudaniel/2cba778ea7135c66e3acaa6fbbab7187.js"></script>

#### Numeric Feature Columns
For Numeric Column we can make use of `numeric_column` for each continuous feature colums.

<script src="https://gist.github.com/jinudaniel/91773e3c3626d31a891a004223197837.js"></script>

#### Building the Logistic Regression Model
Before building the model we will first need to define an input function. The `input_fn` is used to pass feature and labels to the train, evaluate and predict methods of the Estimator.

<script src="https://gist.github.com/jinudaniel/641b6a4264ce4d9617548c6d38bcb197.js"></script>

#### Train and Evaluate
Training a model is just a single command using the tf.estimator API
<script src="https://gist.github.com/jinudaniel/28a60eab3ab9b214a27a11648f7e1506.js"></script>

We can evaluate the model’s accuracy using the `evaluate()` function, using our test data set for validation.
<script src="https://gist.github.com/jinudaniel/9212751eb1fda4d787038702d46ebeb5.js"></script>

Accuray on the test set is 0.836174 i.e. 83.62%.

### Further Reading
- [Introduction to TensorFlow Datasets and Estimators](https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html){:target="_blank"}
- [Estimators](https://www.tensorflow.org/programmers_guide/estimators){:target="_blank"}