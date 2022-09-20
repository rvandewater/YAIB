source activate yaib_updated
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m icu_benchmarks.run train \
                                -c configs/ricu/Classification/TCN.gin \
                                -l logs/benchmark_exp/TCN/ \
                                -t Mortality_At24Hours \
                                -o True \
                                --hidden 256 \
                                -lr 1e-4 \
                                --do 0.0 \
                                --kernel 4 \
                                -sd 1111 2222 3333 4444 5555 6666 7777 8888 9999 0000
