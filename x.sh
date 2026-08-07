set -- hal 0 hal 1 hal 2 hal 3    glados 0 glados 1 glados 2 glados 3    gh 0 hg 0
n=25
while :
do case $# in
       0) break ;;
       *) h=$1; shift
	  i=$1; shift
	  export h
	  ./hal.sh env CUDA_VISIBLE_DEVICES=$i nvcc -I. -arch native launch.cu -o launch.exe '&&' ./launch.exe -o $h.$i -n $((1<<n)) &&
	      ./hal.sh -f $h.$i.raw $h.$i.meta . &&
	      python3 launch.py $h.$i > data/$h.$i
	  ;;
   esac
done
