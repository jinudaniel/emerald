---
title: TensorFlow with TensorBoard
comments: true
---

This post assumes that you know the basics of TensorFlow. If you are not aware of the basics, check out the [lecture notes](https://cs224d.stanford.edu/lectures/CS224d-Lecture7.pdf){:target="_blank"} from Stanford.

### Introduction
TensorBoard is graph visualization software included with any standard TensorFlow installation. TensorBoard helps engineers to analyze, visualize, and debug TensorFlow graphs. Learning to use TensorBoard early and often will make working with TensorFlow much more productive.  
TensorBoard, when fully configured, will look something like this. Image from TensorBoard’s website.

![TensorBoard](/img/tensorboard_1.png "TensorBoard")
To see a TensorBoard in action, click [here](https://www.tensorflow.org/get_started/graph_viz){:target="_blank"}.

### Launching TensorBoard
Lets write a simple TensorFlow program and visualize with TensorBoard. We will make use of `tf.summary.FileWriter` to create an event file in a given directory and add summaries and events to it.

{% highlight python linenos %}
import tensorflow as tf

a = tf.add(1, 2)
b = tf.add(3, 4)

c = tf.multiply(a, b)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graph', sess.graph)
    sess.run(c)
    writer.close()
{% endhighlight %}
Now go to Terminal or command prompt and run the below line. Make sure that your present working directory is the
same as where you ran your Python code.
{% highlight shell %}
tensorboard --logdir="./graph"
{% endhighlight %}
Open your browser and go to [http://localhost:6006/](http://localhost:6006/) (or the link you get back after running
tensorboard command) to open TensorBoard.

![TensorBoard 2](/img/tensorboard_2.png)


### Adding names
You can use the name property to give a meaningful name to the operation. This will help us in better visualizing the operation in TensorBoard.

{% highlight python linenos %}
a = tf.add(1, 2, name='First_Add')
b = tf.add(3, 4, name='Second_Add')

c = tf.multiply(a, b, name='Final_Result')

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graph', sess.graph)
    sess.run(c)
    writer.close()
{% endhighlight %}
![TensorBoard 3](/img/tensorboard_3.png "Adding Names")

### Creating Scopes
Typical TensorFlow graphs can have many thousands of nodes--far too many to see easily all at once, or even to lay out using standard graph tools. To simplify, variable names can be scoped and the visualization uses this.  
A scope can be defined by using `tf.name_scope`.
{% highlight python linenos %}
with tf.name_scope('Operation_A'):
    a1 = tf.add(1, 2, name='First_A_Add')
    a2 = tf.add(3, 4, name='Second_A_Add')
    a3 = tf.multiply(a1, a2, name ='Multiply_A')
with tf.name_scope('Operation_B'):
    b1 = tf.add(5, 6, name='First_B_Add')
    b2 = tf.add(7, 8, name='Second_B_Add')
    b3 = tf.multiply(b1, b2, name ='Multiply_B')
    

c = tf.multiply(a3, b3, name='Final_Result')

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graph', sess.graph)
    print(sess.run(c))
    writer.close()
{% endhighlight %}
![TensorBoard 4](/img/tensorboard_4.png "Adding Scope")

If you want to dive deeper, start by watching this video from the 2017 TensorFlow Developers Conference or you can through the [official documentation](https://www.tensorflow.org/get_started/summaries_and_tensorboard){:target="_blank"}.
<iframe width="560" height="315" src="https://www.youtube.com/embed/eBbEDRsCmv4" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen> </iframe>
