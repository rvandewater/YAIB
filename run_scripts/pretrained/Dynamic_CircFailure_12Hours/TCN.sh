source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Classification/TCN.gin \
                             -l files/pretrained_weights/TCN/ \
                             -t Dynamic_CircFailure_12Hours\
                             -o True \
                             --hidden 256 \
                             -lr 3e-4\
                             --do 0.1 \
                             --kernel 4 \


