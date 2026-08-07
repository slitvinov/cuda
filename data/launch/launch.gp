set grid
set xlabel "useconds"
set key bottom
plot [6:12] for [f in "glados.0 glados.1 glados.2 glados.3 hal.0 hal.1 hal.2 hal.3 gh.0 hg.0 b200"] \
     f using ($1/1e3):3 with steps lw 2 title f
set terminal pdf
set output "launch.pdf"
replot
