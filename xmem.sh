[ $# -eq 0 ] && set -- hal 0 hal 1 hal 2 hal 3    glados 0 glados 1 glados 2 glados 3    gh 0 hg 0
n=16384
s=$((1<<26))
while :
do case $# in
       0) break ;;
       *) h=$1; shift
	  i=$1; shift
	  export h
	  ./hal.sh env CUDA_VISIBLE_DEVICES=$i nvcc -I. -arch native mem.cu -o mem.exe '&&' ./mem.exe -o $h.$i -n $n -s $s &&
	      ./hal.sh -f $h.$i.raw $h.$i.meta . &&
	      python3 mem.py $h.$i > data/mem/$h.$i
	  ;;
   esac
done
