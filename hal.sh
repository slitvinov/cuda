#!/bin/sh
case $# in
  0) payload=$(cat) ;;
  *) payload=$* ;;
esac
b64=$(printf '%s' "$payload" | base64 | tr -d '\n')
rsync max.cu hal: &&
  ssh hal "sh -lc \"\$(echo $b64 | base64 -d)\""
