source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Classification/transformer.gin \
                             -l files/pretrained_weights/transformer/ \
                             -t Mortality_At24Hours\
                             --maxlen 288 \
                             -o True \
                             -lr 1e-5\
                             --hidden 256 \
                             --do 0.1 \
                             --depth 2 \
                             --heads 4 \


