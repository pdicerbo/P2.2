CXX = g++
CC = gcc
SRC = laplace.c
MY_LIBS = random_gen.o system_solvers.o laplace_utils.o
LIBS=-lm
CXXFLAGS = -Wall -std=c++11 -O2 
CFLAGS = -Wall -O2
OBJ = $(SRC:%.c=%.o)

all: default

default: ../laplace.x

../laplace.x : $(OBJ) $(MY_LIBS)
	$(CXX) -o $@ $(LIBS) $^

laplace.o: laplace.c
	$(CXX) $(CXXFLAGS) -c $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $^

%.o: %.c
	$(CC) $(CFLAGS) -c $^

clean:
	rm -rf *.o ../laplace.x
