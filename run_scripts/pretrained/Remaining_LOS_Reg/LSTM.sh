source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Regression/LSTM.gin \
                             -l files/pretrained_weights/LSTM/ \
                             -t Remaining_LOS_Reg\
                             -o True \
                             -lr 3e-4\
                             --hidden 256 \
                             --do 0.2 \
                             --depth 3 \


