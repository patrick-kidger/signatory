.. currentmodule:: info

What is the signature transform?
--------------------------------
If you're reading this documentation then it's probably because you already know what the signature transform is, and are looking to use it in your project. But in case you've stumbled across this and are curious what all the fuss is about...

The 'signature transform' is a transformation that does a particularly good job extracting features from streams of data. Check out `this <https://arxiv.org/abs/1603.03788>`_ for a primer on its use in machine learning, as a feature transformation. Then have a look `here <https://arxiv.org/abs/1905.08494>`_ for a more in-depth look at building it into neural network models, as an arbitrary layer anywhere within a neural network. It's pretty cool.

In brief: the signature of a path determines the path essentially uniquely, and does so in an efficient, computable way.  Furthermore, the signature is rich enough that every continuous function of the path may be approximated arbitrarily well by a linear function of its signature; it is what we call a ‘universal nonlinearity’. Now for various reasons this is a mathematical idealisation not borne out in practice (which is why we put them in a neural network and don't just use a simple linear model), but they still work very well!