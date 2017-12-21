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
Great! Now you can start writing your first program in Python. Fire up IDLE that was installed as part of Python installation and write the below code

{% highlight python linenos %}

print('Hello World !!')
{% endhighlight %}

This print Hello World!! To the console.

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
Subsets of strings can be taken using the slice operator ([ ] and [:] ) with indexes starting at 0 in the beginning of the string and working their way from -1 at the end.

{% highlight python linenos %}
str = 'Hello World!'

print(str)         	 # Prints complete string
print(str[0])      	 # Prints first character of the string
print(str[2:5])    	 # Prints characters starting from 3rd to 5th
print(str[2:])       # Prints string starting from 3rd character
print(str * 2 )    	 # Prints string two times
print(str + "Again") # Prints concatenated string
{% endhighlight %}



