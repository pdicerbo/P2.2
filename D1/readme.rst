D1 Assignment
===============

This folder contains the results of the day-1 assignment of the P2.2 course.

In the **src** folder there are the sources files subdivided in the following way:

- **random-gen.cpp** is the provided C++ code with which fill the matrices and so on.

- **system_solvers.c** is the C library that I write in order to complete the assignment.

- **main.c** is the main file, in which there are the calls to the functions in system_solvers.c;
  the correspondent executable will write some files in **results** folder.

- **plot.py** is a Python script that produces the required plots (saved in **results** folder).

In **D1** folder, do **make** will produce the executable **main.x** that aims to answer at all the questions in the assignment.
Do **make test** will run some tests following the instructions within test/test.c.
Do **make plot** will run the **src/plot.py** script that produces the required plots and store it in **results** folder.
Do **make clean** will clean the produced files.
