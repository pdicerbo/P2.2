default: main

main:
	$(MAKE) -C src

test:
	$(MAKE) -C $@

plot:
	python src/plot.py

x: main
	./main.x
	./laplace.x
	./guess.x

clean:
	$(MAKE) -C src clean
	$(MAKE) -C test clean

.PHONY: test
