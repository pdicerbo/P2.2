CXX = g++
CC = gcc
SRC = main.c
MY_LIBS = random_gen.o system_solvers.o laplace_utils.o
LIBS=-lm
CXXFLAGS = -Wall -std=c++11 -O2 
CFLAGS = -Wall -O2
OBJ = $(SRC:%.c=%.o)

all: default

default: ../main.x
default: ../laplace.x
default: ../guess.x

../main.x : $(OBJ) $(MY_LIBS)
	$(CXX) -o $@ $(LIBS) $^

../laplace.x : laplace.o $(MY_LIBS)
	$(CXX) -o $@ $(LIBS) $^

../guess.x : guess.o $(MY_LIBS)
	$(CXX) -o $@ $(LIBS) $^

laplace.o: laplace.c
	$(CXX) -c $^

guess.o: guess.c
	$(CXX) -c $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $^

%.o: %.c
	$(CC) $(CFLAGS) -c $^

clean:
	rm -rf *.o ../main.x ../laplace.x ../guess.x
