CXX = g++
CC = gcc
SRC = laplace.c
MY_LIBS = random_gen.o system_solvers.o laplace_utils.o
LIBS= #-lm -fopenmp
CXXFLAGS = -Wall -O2 #-std=c++11 -O2
CFLAGS = -Wall -O2
OBJ = $(SRC:%.c=%.o)

all: default

default: ../laplace.x
default: ../benchmark.x

parallel: CXXFLAGS += -D__MPI -D__VECTORIZE #-no-vec -vec-report
parallel: CFLAGS += -D__MPI -D__VECTORIZE #-no-vec -vec-report
parallel: CXX = mpicxx
parallel: CC = mpicc
parallel: default

../laplace.x : $(OBJ) $(MY_LIBS)
	$(CXX) -o $@ $(LIBS) $^

../benchmark.x : benchmark.o $(MY_LIBS)
	$(CXX) -o $@ $(LIBS) $^

laplace.o: laplace.c
	$(CXX) $(CXXFLAGS) -c $^

benchmark.o: benchmark.c
	$(CXX) $(CXXFLAGS) -c $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $^

%.o: %.c
	$(CC) $(CFLAGS) -c $^

clean:
	rm -rf *.o ../laplace.x ../benchmark.x
