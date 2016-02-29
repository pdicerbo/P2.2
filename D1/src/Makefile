CXX = g++
CC = gcc
SRC = main.c
MY_LIBS = random_gen.o system_solvers.o
LIBS=-lm
CXXFLAGS = -Wall -std=c++11 -O2 
CFLAGS = -Wall -O2
OBJ = $(SRC:%.c=%.o)

all: default

default: ../main.x

../main.x : $(OBJ) $(MY_LIBS)
	$(CXX) -o $@ $(LIBS) $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $^

%.o: %.c
	$(CC) $(CFLAGS) -c $^

clean:
	rm -rf *.o ../main.x
