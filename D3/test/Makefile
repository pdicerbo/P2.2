CXX = g++
CC = gcc
VPATH = ../src
SRC = test.c
MY_LIBS = random_gen.o system_solvers.o
LIBS=-lm
CXXFLAGS = -Wall -std=c++11 -O2 
CFLAGS = -Wall -O2
OBJ = $(SRC:%.c=%.o)

all: default

default: test.x
	./test.x
	$(MAKE) -C ../src/ clean
	rm -rf *.o test.x

test.x : $(OBJ) $(MY_LIBS)
	$(CXX) -o $@ $(LIBS) $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $^

%.o: %.c
	$(CC) $(CFLAGS) -c $^

clean:
	rm -rf *.o test.x
