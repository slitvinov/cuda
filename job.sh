#!/bin/sh
read j < job
case $# in
  0) echo "usage: $0 [-f files ...] | command ..." >&2; exit 2 ;;
esac
out=${TMPDIR:-/tmp}/out.$$
aux=${TMPDIR:-/tmp}/aux.$$
trap 'rm -f "$out" "$aux"' 0
trap 'exit 1' 1 2 15
ssh rc "srun --overlap -n 1 --jobid $j hostname" > "$out"
read h < "$out"
n=${h%%.*}
case $1 in
  -f)
    shift
    for f; do printf '%s\n' "$f"; done > "$aux"
    rsync --files-from="$aux" "$h:/tmp/" .
    exit
    ;;
esac
b64=$(printf 'cd /tmp && %s' "$*" | base64 | tr -d '\n')
rsync *.h *.cu "$h:/tmp/" &&
  ssh -n rc "srun --chdir=/tmp --overlap --jobid $j -w $n sh -lc \"ml cuda && \$(echo $b64 | base64 -d)\""
