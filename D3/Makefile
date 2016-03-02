default: parallel #main

main:
	$(MAKE) -C src

test:
	$(MAKE) -C $@

parallel:
	$(MAKE) -C src parallel
plot:
	python src/plot.py

x: main
	./laplace.x

clean:
	$(MAKE) -C src clean
	$(MAKE) -C test clean

.PHONY: test
