#!/bin/bash

# Run using the following command:
# { ./bench.sh; } > out.txt 2>&1
# to output everything to out.txt.

echo "1"
time ./multiply 1 1 1 1
echo "--------------------------"

for number in {16000..16000..500}
do
echo "$number"
time ./multiply $number $number $number $number
echo "--------------------------"
done
exit 0
