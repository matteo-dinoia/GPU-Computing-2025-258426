#!/usr/bin/bash
echo 'COMPILING'
make normal || exit;

echo 'RUNNING (may take some time to finish)';
./build/main $1