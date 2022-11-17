#!/bin/bash

eval names=$4
eval source_dirs=$5
for var in ${names[@]}
do
    for i in ${!names[@]}
    do
        if [ $var != ${names[$i]} ];
        then
            icu-benchmarks evaluate \
                -d $2$var \
                -n $var \
                -e $1 \
                -sn ${names[$i]} \
                --source-dir ${source_dirs[$i]} \
                -l $3 \
                -c \
                -s 1111 2222 3333 4444 5555
        fi
    done
done

        