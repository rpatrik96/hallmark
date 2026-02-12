.PHONY: all clean

all:
	latexmk -pdf main.tex

clean:
	latexmk -C
	rm -f *.bbl *.synctex.gz

watch:
	latexmk -pdf -pvc main.tex
