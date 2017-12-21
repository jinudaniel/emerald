---
title: Introduction to Python
---

### What is Python
Python is a general purpose programming language created by Guido Van Rossum. Python is most praised for its elegant syntax and readable code. If you are just beginning your programming career, python suits you best.  
With python you can do everything from GUI development, Web application, System administration tasks, Financial calculation, Data Analysis, Visualization and list goes on.

### Installing Python:
You can head over to the below link to download python. My recommendation would be to install Python version > 3.6 as Python 2.7 will become obsolete by 2020

[Download](https://www.python.org/downloads/)

### Your first program:
Great! Now you can start writing your first program in Python. Fire up the IDLE that was installed as part of Python installation and write the below code

{% highlight python linenos %}

print('Hello World !!')
{% endhighlight %}

This prints Hello World !! to the console.

### Basic Data Types:
Like most languages, Python has a number of basic types including integers, floats, booleans, and strings. These data types behave in ways that are familiar from other programming languages.

#### Numbers:
Number data types store numeric values.

{% highlight python linenos %}
x = 5
print(type(x))  # prints <class 'int'>
print(x + 1)    # Addition
print(x - 1)	# Subtraction
print(x * 3)	# Multiplication
print(x / 2)	# Division
print(x ** 2)	# Exponentiation
{% endhighlight %}

#### Boolean : 
Python implements all of the usual operators for Boolean logic, but uses English words rather than symbols (&&, ||, etc.):

{% highlight python linenos %}
t = True
f = False
print(type(t))    # prints <class 'bool'>
print(t and f)    # Logical AND. Prints "False"
print(t or f)	  # Logical OR. Prints "True"
print(not t)	  # Logical NOT. Prints "False"
{% endhighlight %}

#### Strings:
Strings in Python are identified as a contiguous set of characters represented in the quotation marks.  
Python allows for either pairs of single or double quotes.  
Subsets of strings can be taken using the slice operator ([[:]) with indexes starting at 0 in the beginning of the string and working their way from -1 at the end.

{% highlight python linenos %}
str = 'Hello World!'

print(str)         	 # Prints complete string
print(str[0])      	 # Prints first character of the string
print(str[2:5])    	 # Prints characters starting from 3rd to 5th
print(str[2:])       # Prints string starting from 3rd character
print(str * 2 )    	 # Prints string two times
print(str + "Again") # Prints concatenated string
{% endhighlight %}

### List
List is an ordered sequence of items. It is one of the most used datatype in Python and is very flexible. All the items in a list do not need to be of the same type.  
Declaring a list is pretty straight forward. Items separated by commas are enclosed within brackets [ ].  
Lists are mutable, meaning, value of elements of a list can be altered.
{% highlight python linenos %}
list = [ 'john', 123 , 9930, 'doe', 70.2 ]
anotherlist = [500, 'jane']

print(list)         # Prints complete list
print(list[0])       # Prints first element of the list
print(list[1:3])     # Prints elements starting from 2nd till 3rd 
print(list[2:])      # Prints elements starting from 3rd element
print anotherlist * 2  # Prints list two times
print list + anotherlist # Prints concatenated lists
{% endhighlight %}

### Tuple

A tuple is an (immutable) ordered list of values.  
The main differences between lists and tuples are: Lists are enclosed in brackets [ ] and their elements and size can be changed, while tuples are enclosed in parentheses ( ) and cannot be updated.  
{% highlight python linenos %}
t = (2, 3)
print(t)	# Prints the tuple.
print(t[0])	# Prints the first elemnt of the tuple
{% endhighlight %}

### Dictionary
Dictionary is an unordered collection of key-value pairs.  
Dictionaries are optimized for retrieving data. We must know the key to retrieve the value.  

In Python, dictionaries are defined within braces {} with each item being a pair in the form key:value. Key and value can be of any type.
{% highlight python linenos %}
d = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data.
print(d['cat'])	# Get an entry from the dictionary using the key. Prints cute.
d['mouse'] = 'small'	# Add an entry to the dictionary.
print(d['mouse'])		# Prints small.
del d['mouse']			# Remove an entry from the dictionary.
{% endhighlight %}



