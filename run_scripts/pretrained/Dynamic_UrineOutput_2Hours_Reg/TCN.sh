source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Regression/TCN.gin \
                             -l files/pretrained_weights/TCN/ \
                             -t Dynamic_UrineOutput_2Hours_Reg\
                             -o True \
                             --hidden 256 \
                             -lr 3e-4\
                             --do 0.2 \
                             --kernel 8 \


