# PTA and Reduced Order Quadrature v2

This is another take at the reduced order quadrature and pulsar timing
analysis.  By starting this project again I am mainly interested in seeing if
it possible to get any useful code (which would work for simple and not so
simple cases).

I will approach this from a test driven development point of view which will
hopefully be more productive.

All the code should be written in C++ and compiled with Clang because I want to
learn the language.  Tests will be written with the
[CATCH](https://github.com/philsquared/Catch) framework.  Since this is
implemented only in headers, I do not need to install it in the `sudo make
install` sense, but I will need to clone it from the `github` repository.

## The structure and dependencies

Since I am doing this the second time, I will need to plan it better so that
I have a writeup straight away, before even having a line of code.

### General work flow

* Write Python prototype code with unit-tests in the TDD way.

* Use `numpy` and so on.

* Once the code works (generates RB, and does EIM), add a bit of Cython.

* Then if it is too slow, reimplement it in C++ and use the Python/Cython code
to generate results for testing.  Save some calculations into files and then
compare them with the ones I get from the C++ code.

### C++ dependencies

I want to use a proper linear algebra package, because they are more optimized.
It looks that the [TNT headers](http://math.nist.gov/tnt/download.html) are
quite good.  The best thing is that everything is defined in the header level.

I also need to use [MultiNest](https://github.com/JohannesBuchner/MultiNest)
for the actual Bayesian parameter inference.  But before I start doing that
I need to have a working greedy basis generator, which needs to be implemented
in an OO way.

