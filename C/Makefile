CC=clang
CFLAGS=--std=c99 -Wall

main: ans.o nanotorch.o
	$(CC) $(CFLAGS) -o main main.c ans.o nanotorch.o 

#arch.o cifar-10-fc-runtime.o

nanotorch.o: nanotorch.c nanotorch.h
	$(CC) $(CFLAGS) -c nanotorch.c

ans.o: ans.c ans.h
	$(CC) $(CFLAGS) -c ans.c

#arch.o: arch.h
#	$(CC) $(CFLAGS) -c arch.h

#cifar-10-fc-runtime.o: cifar-10-fc-runtime.h
#	$(CC) $(CFLAGS) -c cifar-10-fc-runtime.h

clean:
	rm -f main
	rm -f *.o
	rm -f *.h.gch