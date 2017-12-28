---
title: Data Wrangling with Pandas
comments: true
---

> Data science is 90% cleaning the data and 10% complaining about cleaning the data.  

 
Data wrangling is an essential part of any data analysis. Before any algorithms are applied to the data set it is crucial that the data is checked and ready for consumption.  
For example, if your data set is incomplete, or has null values, the analysis will not be complete and correct.  
**Pandas** is a high-level data manipulation tool developed by Wes McKinney. It is built on the Numpy package and its key data structure is called the DataFrame. DataFrames allow you to store and manipulate tabular data in rows of observations and columns of variables.  
### DataFrame

#### Creating a DataFrame
{% highlight python %}
import numpy as np
import pandas as pd
from numpy.random import randn
np.random.seed(101)

df = pd.DataFrame(randn(5,4), index=['A','B','C','D','E'], columns=['W', 'X', 'Y', 'Z'])
print(df)
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>0.302665</td>
      <td>1.693723</td>
      <td>-1.706086</td>
      <td>-1.159119</td>
    </tr>
    <tr>
      <th>B</th>
      <td>-0.134841</td>
      <td>0.390528</td>
      <td>0.166905</td>
      <td>0.184502</td>
    </tr>
    <tr>
      <th>C</th>
      <td>0.807706</td>
      <td>0.072960</td>
      <td>0.638787</td>
      <td>0.329646</td>
    </tr>
    <tr>
      <th>D</th>
      <td>-0.497104</td>
      <td>-0.754070</td>
      <td>-0.943406</td>
      <td>0.484752</td>
    </tr>
    <tr>
      <th>E</th>
      <td>-0.116773</td>
      <td>1.901755</td>
      <td>0.238127</td>
      <td>1.996652</td>
    </tr>
  </tbody>
</table>
</div>


