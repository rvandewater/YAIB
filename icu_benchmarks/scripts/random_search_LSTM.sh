#!/bin/bash
for i in {1..25}
do
   python -m icu_benchmarks.run train \
    -d $1 \
    -n hirid \
    -t Mortality_At24Hours \
    -m LSTM \
    -hp DLWrapper.train.patience=5
    -c \
    -s 1111 2222 3333
done