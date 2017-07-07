#!/bin/bash

# Run using the following command:
# { ./bench.sh; } > out.txt 2>&1
# to output everything to out.txt.

get_multiple()
{
    NUMBER=$1
    MULTIPLE=$2
    shift; shift;

    divided=$(($NUMBER/$MULTIPLE))

    if [ $(($NUMBER % $MULTIPLE)) -ne 0 ]
    then
        divided=$(($divided + 1))
    fi

    multiple=$(($divided * $MULTIPLE))
}

get_multiple 1 32

echo "$multiple"
time ./multiply $multiple $multiple $multiple $multiple
echo "--------------------------"

for number in {500..16000..500}
do
    get_multiple $number 32

    echo "$multiple"
    time ./multiply $multiple $multiple $multiple $multiple
    echo "--------------------------"
done
exit 0

