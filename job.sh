#!/bin/sh
read j < job
case $# in
  0) payload=$(cat) ;;
  *) payload=$* ;;
esac
out=${TMPDIR:-/tmp}/out.$$
trap 'rm -f "$out"' 0
trap 'exit 1' 1 2 15

b64=$(printf 'cd /tmp && %s' "$payload" | base64 | tr -d '\n')
ssh rc "srun --overlap -n 1 --jobid $j hostname" > "$out"
read h < "$out"
n=${h%%.*}
rsync *.h *.cu "$h:/tmp/" &&
ssh rc "srun --chdir=/tmp --overlap --jobid $j -w $n sh -lc \"ml cuda && \$(echo $b64 | base64 -d)\""
