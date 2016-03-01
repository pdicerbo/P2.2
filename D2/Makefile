default: main

main:
	$(MAKE) -C src

test:
	$(MAKE) -C $@

plot:
	python src/plot.py
clean:
	$(MAKE) -C src clean
	$(MAKE) -C test clean

.PHONY: test
