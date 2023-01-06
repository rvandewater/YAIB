source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Classification/transformer.gin \
                             -l files/pretrained_weights/transformer/ \
                             -t Phenotyping_APACHEGroup --loss-weight balanced \
                             --num-class 15 \
                             --maxlen 288 \
                             -o True \
                             -lr 3e-4\
                             --hidden 32 \
                             --do 0.2 \
                             --depth 2 \
                             --heads 8 \


