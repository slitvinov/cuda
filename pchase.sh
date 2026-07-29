set -eu
mkdir -p data2
s=0
while test $s -lt 132
do ./hal.sh nvcc -I. -arch native -run pchase.cu -run-args -s,$s,-i,15000,-n,$((1<<14)),-w,64,-b,x.$s
   case $? in
       0 ) ;;
       * ) exit $? ;;
   esac
   ./hal.sh -f x.$s.raw x.$s.meta data2/
   case $? in
       0 ) ;;
       * ) exit $? ;;
   esac
   s=$((s+1))
done
