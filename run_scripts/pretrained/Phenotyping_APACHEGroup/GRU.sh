source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Classification/GRU.gin \
                             -l files/pretrained_weights/GRU/ \
                             -t Phenotyping_APACHEGroup --loss-weight balanced \
                             --num-class 15 \
                             --maxlen 288 \
                             -o True \
                             -lr 1e-5 \
                             --hidden 256 \
                             --do 0.4 \
                             --depth 1 \


