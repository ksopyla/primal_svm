# Primal SVM implementation in python

This python port of Primal SVM - fast linear SVM implementation by Olivier Chapelle http://olivier.chapelle.cc/primal/

There are two solvers implemented:

* Newton method solver - Newton solver is better suited for problems with large number of examples (>1M) but small dimension (less than 1000)
* Conjugate Gradient solver - better for large and sparse problems


 