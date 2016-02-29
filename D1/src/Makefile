CXX = g++
CC = g++
SRC = main.c
MY_LIBS = random_gen.o
LIBS=-lm
CXXFLAGS = -Wall -O2 -std=c++11
CFLAGS = -Wall -O2
OBJ = $(SRC:%.c=%.o)

all: default

default: ../main.x

../main.x : $(OBJ) $(MY_LIBS)
	$(CC) -o $@ $(LIBS) $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $^

%.o: %.c
	$(CC) $(CFLAGS) -c $^

clean:
	rm -rf *.o ../main.x
