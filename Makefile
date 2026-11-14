CC=mpicc
CFLAGS=-O3 -std=c11 -Wall -Wextra -fopenmp
LDLIBS=-lm -fopenmp
all: fire channel deadlock
fire: src/fire.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
deadlock: src/deadlock.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

channel: src/channel.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
clean:
	rm -f fire channel deadlock
