source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Classification/TCN.gin \
                             -l files/pretrained_weights/TCN/ \
                             -t Mortality_At24Hours \
                             --maxlen 288 \
                             -o True \
                             --hidden 256 \
                             -lr 1e-4\
                             --do 0.0 \
                             --kernel 4 \


