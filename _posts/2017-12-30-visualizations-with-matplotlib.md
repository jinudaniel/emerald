---
title: Visualizations with Matplotlib
comments: true
---

### What is Visualization?
A human mind can easily read and understand a chart or image as compared to looking through a large chunk of data in a table or a spreadsheet. Data visualization is a powerful technique to visualize and get meaningful insights from the dataset. 

### Matplotlib
Matplotlib provides a way to easily generate a wide variety of plots and charts in a few lines of Python code. It is an open source project that can be integrated into Python scripts, jupyter notebooks, web application servers, and multiple GUI toolkits.  

#### Installing Matplotlib
The best way to install matplotlib is by downloading the [Anaconda](https://www.anaconda.com/download/) distribution. Anaconda is a set of python libraries which has the standard python programming language libraries as well as numerous third party scipy libraries like numpy, pandas, matplotlib etc.  
Or you can install matplotlib by using `pip`
{% highlight python %}
pip install matplotlib
{% endhighlight %}

#### Plotting a simple graph.
Let's start by plotting a simple graph.
{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline  # Add this line if you are using Jupyter notebook.
{% endhighlight %}

{% highlight python %}
x = np.linspace(0,5,11)
y = x**2
{% endhighlight %}

{% highlight python %}
plt.plot(x,y)
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Sample plot')
{% endhighlight %}

![Sample Plot](/img/matplotlib_1.png "Sample plot")

#### Adding a legend
A proper figure is not complete without its own legend. Matplotlib provides a way to generate a legend with the minimal amount of effort. 

{% highlight python %}
fig = plt.figure()
axes = fig.add_axes([0,0,1,1])
axes.plot(x, x**2, label='X Squared')
axes.plot(x,x**3, label = 'X Cubed')
axes.legend()
{% endhighlight %}

![Legend](/img/matplotlib_2.png "Legend")

#### Creating a Subplot
Sometimes it is helpful to compare different views of data side by side. To help with this, Matplotlib has the concept of Subplots: groups of smaller axes that can exist together within a single figure.  

{% highlight python %}
fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].plot(x,y)
axes[1].plot(y,x)
{% endhighlight %}

![Subplot](/img/matplotlib_3.png "Subplot")

#### Customizing Plot Appearance

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
{% endhighlight %}

{% highlight python %}
x = np.linspace(0,5,11)
y = x**2
{% endhighlight %}

{% highlight python %}
plt.plot(x,y, color = 'red') # add color to the plot
{% endhighlight %}

![Add color](/img/matplotlib_4.png "Add color")

{% highlight python %}
# add line style and line width to the plot
plt.plot(x,y,color ='purple', linewidth = 3, linestyle = '--') 
{% endhighlight %}

![Add line style and width](/img/matplotlib_5.png "Add line style and width")

{% highlight python %}
# add marker for the points on the plot.
plt.plot(x,y,color ='purple', linewidth = 3, linestyle = '--', marker = 'o') 
{% endhighlight %}

![Add marker](/img/matplotlib_5.png "Add marker")


