CC = gcc
CFLAGS = -Wall -std=c99 -c -g -O3 -lpthread
LFLAGS = -Wall -std=c99 -lpthread

multiply: multiply.o
	$(CC) $(LFLAGS) multiply.o -o multiply

multiply.o: multiply.c ../matrix.c ../matrix.h
	$(CC) $(CFLAGS) multiply.c
