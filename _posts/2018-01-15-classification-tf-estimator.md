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
{% highlight python linenos %}
import pandas as pd

df = pd.read_csv('census_data.csv')
{% endhighlight %}

Since the task is Binary Classification, we will make use of Pandas `apply` function to convert the `income_bracket` column to a label whose value is 1 if the income is above $50k and 0 otherwise.
{% highlight python linenos %}
def fix_label(label):
    if label == ' <=50K':
        return 0
    else:
        return 1

df['income_bracket'] = df['income_bracket'].apply(fix_label)
{% endhighlight %}

#### Train and Test Split.
Let's make use of sklearn's train_test_split method to split the data into training and test set.
{% highlight python linenos %}
from sklearn.model_selection import train_test_split
X = df.drop('income_bracket', axis = 1)
Y = df['income_bracket']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
{% endhighlight %}

#### Categorical Feature Columns
Feature column is an abstract concept of any raw or derived variable that can be used to predict the target label. To define a feature column for a categorical feature, we can create a `CategoricalColumn` using the tf.feature_column API.    
If we know the possible values for a Categorical feature columns then we can make use of `categorical_column_with_vocabulary_list` but if we are not aware of all the possible values then we can make use of `categorical_column_with_hash_bucket` with the hash_bucket_size defined.

{% highlight python linenos %}
workclass = tf.feature_column.categorical_column_with_hash_bucket('workclass', hash_bucket_size=10)
education = tf.feature_column.categorical_column_with_hash_bucket('education', hash_bucket_size=100)
marital_status = tf.feature_column.categorical_column_with_hash_bucket('marital_status', hash_bucket_size=100)
occupation = tf.feature_column.categorical_column_with_hash_bucket('occupation', hash_bucket_size=100)
relationship = tf.feature_column.categorical_column_with_hash_bucket('relationship', hash_bucket_size=100)
race = tf.feature_column.categorical_column_with_hash_bucket('race', hash_bucket_size=100)
gender = tf.feature_column.categorical_column_with_hash_bucket('gender', hash_bucket_size=100)
native_country = tf.feature_column.categorical_column_with_hash_bucket('native_country', hash_bucket_size=100)
{% endhighlight %}

#### Numeric Feature Columns
For Numeric Column we can make use of `numeric_column` for each continuous feature colums.

{% highlight python linenos %}
age = tf.feature_column.numeric_column('age')
education_num = tf.feature_column.numeric_column('education_num')
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')
{% endhighlight %}

#### Building the Logistic Regression Model
Before building the model we will first need to define an input function. The `input_fn` is used to pass feature and labels to the train, evaluate and predict methods of the Estimator.

{% highlight python linenos %}
feature_columns = [age, workclass, education, education_num, marital_status, occupation, 
				relationship, race, gender, capital_gain, capital_loss, hours_per_week, native_country]

input_func = tf.estimator.inputs.pandas_input_fn(x = X_train, y = y_train, batch_size=128, num_epochs=10, shuffle=False)
model = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=2, model_dir='./output')
{% endhighlight %}

#### Train and Evaluate
Training a model is just a single command using the tf.estimator API
{% highlight python linenos %}
model.train(input_fn=input_func)
{% endhighlight %}

We can evaluate the model’s accuracy using the `evaluate()` function, using our test data set for validation.
{% highlight python linenos %}
eval_fn = tf.estimator.inputs.pandas_input_fn(x = X_test, y=y_test, batch_size=128, shuffle=False)
result = model.evaluate(input_fn=eval_fn)

for key in sorted(result):
  print('%s: %s' % (key, result[key]))
{% endhighlight %}

Accuray on the test set is 0.836174 i.e. 83.62%.

### Further Reading
- [Introduction to TensorFlow Datasets and Estimators](https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html){:target="_blank"}
- [Estimators](https://www.tensorflow.org/programmers_guide/estimators){:target="_blank"}