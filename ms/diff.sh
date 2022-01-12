latexdiff ms_prev.tex ms.tex --no-label --enforce-auto-mbox --type=CFONT > diff.tex
pdflatex diff.tex
pdflatex diff.tex
open diff.pdf