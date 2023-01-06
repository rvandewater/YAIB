source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Classification/LSTM.gin \
                             -l files/pretrained_weights/LSTM/ \
                             -t Dynamic_RespFailure_12Hours \
                             -o True \
                             -lr 1e-4\
                             --hidden 256 \
                             --do 0.4 \
                             --depth 2 \


