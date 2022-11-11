#!/bin/bash
for i in {1..50}
do
   python -m icu_benchmarks.run train \
    -d ../data/mortality_seq/hirid \
    -n hirid \
    -t Mortality_At24Hours \
    -m LGBMClassifier \
    -c \
    -s 1111 2222 3333 4444 5555
done