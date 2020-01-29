CC=		g++

OBJS=		dataset.o utils.o model.o
MAIN=		main
 
all:	$(OBJS) $(MAIN).cpp
	$(CC) -std=c++11 -o $(MAIN) $(MAIN).cpp $(OBJS)
	strip $(MAIN)


dataset.o:	dataset.h dataset.cpp
	$(CC) -std=c++11 -c -o dataset.o dataset.cpp

utils.o:	utils.h utils.cpp
	$(CC) -std=c++11 -c -o utils.o utils.cpp

model.o:	model.h model.cpp
	$(CC) -std=c++11 -c -o model.o model.cpp

test:


clean:
	rm $(OBJS) 
	rm $(MAIN)

