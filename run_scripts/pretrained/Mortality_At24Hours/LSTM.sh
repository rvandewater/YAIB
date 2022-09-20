source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Classification/LSTM.gin \
                             -l files/pretrained_weights/LSTM/ \
                             -t Mortality_At24Hours \
                             --maxlen 288 \
                             -o True \
                             -lr 1e-4\
                             --hidden 128 \
                             --do 0.1 \
                             --depth 1 \


