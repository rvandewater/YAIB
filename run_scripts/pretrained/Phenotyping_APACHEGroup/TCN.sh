source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Classification/TCN.gin \
                             -l files/pretrained_weights/TCN/ \
                             -t Phenotyping_APACHEGroup --loss-weight balanced  \
                             --num-class 15 \
                             --maxlen 288 \
                             -lr 1e-4 \
                             -o True \
                             --hidden 128 \
                             --do 0.2 \
                             --kernel 32 \


