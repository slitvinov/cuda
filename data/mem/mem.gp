set grid
set logscale xy 2
set xlabel "bytes"
set ylabel "useconds"
set key top left
plot [1:1<<26] for [f in "hal.0 hal.1 hal.2 hal.3 glados.0 glados.1 glados.2 glados.3 gh.0 hg.0 b200"] \
     f using 1:($4/1e3) with linespoints lw 2 title f
set terminal pdf
set output "mem.pdf"
replot
