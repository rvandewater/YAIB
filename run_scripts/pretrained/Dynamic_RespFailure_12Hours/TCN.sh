source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Classification/TCN.gin \
                             -l files/pretrained_weights/TCN/ \
                             -t Dynamic_RespFailure_12Hours \
                             -o True \
                             --hidden 64 \
                             -lr 3e-4\
                             --do 0.4 \
                             --kernel 8 \



