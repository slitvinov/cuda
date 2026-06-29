#!/bin/sh
: ${j=22725765}
case $# in
  0) payload=$(cat) ;;
  *) payload=$* ;;
esac
b64=$(printf '%s' "$payload" | base64 | tr -d '\n')
ssh rc srun --jobid "$j" "sh -lc \"\$(echo $b64 | base64 -d)\""
