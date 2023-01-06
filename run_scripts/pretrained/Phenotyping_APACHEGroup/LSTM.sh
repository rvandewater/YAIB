source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Classification/LSTM.gin \
                             -l files/pretrained_weights/LSTM/ \
                             -t Phenotyping_APACHEGroup --loss-weight balanced \
                             --num-class 15 \
                             --maxlen 288 \
                             -o True \
                             -lr 3e-4\
                             --hidden 256 \
                             --do 0.0 \
                             --depth 1 \


