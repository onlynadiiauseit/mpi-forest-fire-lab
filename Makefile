CC=mpicc
CFLAGS=-O3 -std=c11 -Wall -Wextra -fopenmp
LDLIBS=-lm -fopenmp
all: fire channel deadlock
fire: src/fire.c src/profile.h
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
deadlock: src/deadlock.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

channel: src/channel.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
clean:
	rm -f fire channel deadlock .profile_test

test-profile: tests/profile_test.c src/profile.h
	$(CC) $(CFLAGS) -Isrc -o .profile_test $< $(LDLIBS)
	mpirun --oversubscribe -np 2 ./.profile_test
	rm -f .profile_test

.PHONY: all clean test-profile
