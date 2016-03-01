D1 Assignment
===============

This folder contains the results of the day-1 assignment of the P2.2 course.

In **D1** folder:

- Do **make** will produce the executable **main.x** that aims to answer at all the questions in the assignment.

- Do **make test** will run some tests following the instructions within test/test.c.

- Do **make plot** will run the **src/plot.py** script that produces the required plots and store it in **results** folder.

- Do **make clean** will clean the produced files.

In the **src** folder there are the sources files subdivided in the following way:

- **random-gen.cpp** is the provided C++ code with which fill the matrices and so on.

- **system_solvers.c** is the C library that I write in order to complete the assignment.

- **main.c** is the main file, in which there are the calls to the functions in system_solvers.c;
  the main is written in a way such that the correspondent executable will write some output files in **results** folder
  (first_scaling.dat and sec_scaling.dat).

- **plot.py** is a Python script that produces the required plots (saved in **results** folder).

In the **include** folder there are the headers of the libraries.

Content of **src/system_solvers.c**
-------------------------------

The **src/system_solvers.c** contains the function to fulfil the assignment. There are the **gradient_alg** and **conj_grad_alg** functions that
rispectively perform the gradient algorithm and the conjugate gradien algorithm. There is also **minimization_check** function that is used
to perform the check on the minimization of the functional F.
