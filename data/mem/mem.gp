set grid
set logscale xy 2
set xlabel "bytes"
set ylabel "useconds"
set key top left
f(x) = a + b*x
a = 10
b = 0.02
FIT_LIMIT = 1e-12
fit f(x) "b200" using 1:($4/1e3) via a, b
plot [1:1<<26] for [ff in "b200"] ff using 1:($4/1e3) with linespoints lw 2 title ff." median", \
              for [ff in "b200"] ff using 1:($3/1e3) with linespoints lw 2 title ff." min", \
              f(x) lw 2 title sprintf("fit %.1f us + x/(%.1f GB/s)", a, 1e-3/b)
set terminal pdf
set output "mem.pdf"
replot
