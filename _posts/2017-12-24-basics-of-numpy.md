---
title: Basics of Numpy
comments: true
---

Before you start this tutorial you should know a bit about Python. If you dont have any background in Python head over to my [Introduction to Python]({{ site.baseurl }}{% post_url 2017-12-21-Intro-to-Python %}) post.

## Numpy
Numpy or Numerical Python is the core library for scientific computing in Python. It provides a high-performance multidimensional array object, and tools for working with these arrays. 

### Arrays 
A numpy array is a grid of values, all of the same type. The number of dimensions is the rank of the array; the shape of an array is a tuple of integers giving the size of the array along each dimension.

{% highlight python linenos %}
import numpy as np
my_list = [1,2,3]
a = np.array(my_list) # Creates a rank 1 array

my_mat = [[1,2,3][4,5,6][7,8,9]]
b = np.array(my_mat) # Creates a rank 2 array

print(a.shape) # Prints (3,)
print(b.shape) # Prints (3,3)
{% endhighlight %}

Numpy internally provides many functions to create arrays.

{% highlight python linenos %}
np.random.rand(5,5) # Creates a 5x5 matrix of random numbers
					# from a uniform distribution over [0,1]
np.random.randn(3,2) # Creates a 3x2 matrix of random numbers
					 # from a standard normal distribution
np.zeros((3,4)) # Creates a 3x4 matrix of all zeros
np.ones((2,2))  # Creats a 2x2 matrix of all ones
np.eye(4) # Creates a 4x4 Identity matrix
{% endhighlight %}

### Array Indexing
**One Dimensional** arrays can be indexed, sliced and iterated over, much like lists and other Python sequences.
{% highlight python linenos %}
a = np.arange(0,11)
print(a) #Prints array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
print(a[8]) # Prints 8
print(a[0:5]) # Prints array([0, 1, 2, 3, 4])
print(a[6:])  # Prints array([ 6,  7,  8,  9, 10])
{% endhighlight %}

**Multi Dimensional** have one index per axis. These indices are given as a tuple.
{% highlight python linenos %}
a_2d = np.array([[5,10,15],[20,25,30],[35,40,45]])
print(a_2d)
# Prints array([[ 5, 10, 15],
#      			[20, 25, 30],
#      			[35, 40, 45]])

print(a_2d[0][0]) # Bracket notation. Prints 5
print(a_2d[1,2])  # Comma notation. Prints 30
print(a_2d[:2, 1:])
# Prints [[10 15]
#          [25 30]]

{% endhighlight %}
To know more about numpy array indexing you should [read the documentation](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html)

### Numpy operations
Arithmetic operators on arrays apply elementwise. A new array is created and filled with the result.
{% highlight python linenos %}
arr = np.arange(1,6)
print(arr) # Prints [1 2 3 4 5]

#Elementwise Addition
#Prints [ 2  4  6  8 10]
print(arr + arr)

#Elementwise Subtraction
#Prints [0 0 0 0 0]
print(arr - arr)

#Element wise multiplication
#Prints [ 1  4  9 16 25]
print(arr * arr)
{% endhighlight %}

 `*` is elementwise multiplication, not matrix
multiplication. We instead use the `dot` function to compute inner
products of vectors, to multiply a vector by a matrix, and to
multiply matrices. `dot` is available both as a function in the numpy
module and as an instance method of array objects
{% highlight python linenos %}
a = np.array([1,2])
b = np.array([3,4])

#Inner product of vectors. Both prints 11
print(np.dot(a,b))
print(a.dot(b))

x = np.array([[1,2], [3,4]])
y = np.array([[5,6], [7,8]])

#Dot product of the matrices.
#Prints [[19 22]
#        [43 50]]
print(np.dot(x, y))
print(x.dot(y))
{% endhighlight %}

This brief overview has touched on many of the important topics that you need to know about numpy. Check out the documentation to know more about numpy.
[Numpy Reference](https://docs.scipy.org/doc/numpy/reference/)
[Cheat Sheet](http://datacamp-community.s3.amazonaws.com/e6b8c7d1-6e9b-41c5-879f-7f82325cb18f)