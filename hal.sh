#!/bin/sh

: ${h=hg}
case $1 in
  -f)
    shift
    aux=${TMPDIR:-/tmp}/hal.$$
    trap 'rm -f "$aux"' 0
    trap 'exit 1' 1 2 15
    while [ $# -gt 1 ]; do printf '%s\n' "$1"; shift; done > "$aux"
    rsync --timeout=120 --files-from="$aux" $h:/tmp/ "$1"
    exit
    ;;
esac
case $# in
  0) echo "usage: $0 [-f files ...] | command ..." >&2; exit 2 ;;
esac
b64=$(printf '%s' "$*" | base64 | tr -d '\n')
rsync --timeout=120 *.h *.cu *.py $h:/tmp/ &&
  ssh $h "cd /tmp && timeout -k 5 120 sh -lc \"\$(echo $b64 | base64 -d)\""
