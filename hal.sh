#!/bin/sh

if test -r host
then read h < host
else h=gh
fi

case $# in
  0) payload=$(cat) ;;
  *) payload=$* ;;
esac
b64=$(printf '%s' "$payload" | base64 | tr -d '\n')
rsync *.h *.cu $h:/tmp/ &&
  ssh $h "cd /tmp && sh -lc \"\$(echo $b64 | base64 -d)\""
