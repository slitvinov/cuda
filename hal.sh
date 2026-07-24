#!/bin/sh
h=gh
case $# in
  0) payload=$(cat) ;;
  *) payload=$* ;;
esac
b64=$(printf '%s' "$payload" | base64 | tr -d '\n')
rsync *.cu $h:/tmp/ &&
  ssh $h "cd /tmp && sh -lc \"\$(echo $b64 | base64 -d)\""
