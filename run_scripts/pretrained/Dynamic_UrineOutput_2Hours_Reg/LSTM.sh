source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Regression/LSTM.gin \
                             -l files/pretrained_weights/LSTM/ \
                             -t Dynamic_UrineOutput_2Hours_Reg\
                             -o True \
                             -lr 3e-4\
                             --hidden 128 \
                             --do 0.3 \
                             --depth 3 \


