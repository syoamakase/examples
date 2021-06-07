#!/bin/bash

stage=2       # start from -1 if you need to start from data download
stop_stage=100

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
# datadir=/export/a15/vpanayotov/data
datadir=/n/work2/ueno/data/librispeech/data

# base url for downloads.
data_url=www.openslr.org/resources/12


wav_dir='wav'

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data download"
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        local/download_and_untar.sh ${datadir} ${data_url} ${part}
    done
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        # use underscore-separated names in data directories.
        local/data_prep.sh ${datadir}/LibriSpeech/${part} data/${part//-/_}
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Convert flac into wav"
    mkdir -p tmp

    for x in dev_clean; do #test_clean train_clean_100 train_clean_360 train_other_500; do
        cut -d ' ' -f 6 data/${x}/wav.scp > data/${x}/flaclist
        python tools/extract_subdir.py data/${x}/flaclist data/${x}/${wav_dir} -n -2 | sed -e "s/^/mkdir -p /g" | bash
        python tools/make_convert_flac2wav.py data/${x}/flaclist --save_dir data/${x}/${wav_dir} -n -3 > tmp/convert_flac2wav.sh
        bash tmp/convert_flac2wav.sh
        cut -d ' ' -f 4 tmp/convert_flac2wav.sh > data/${x}/wavlist
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Make log mel-scale filter bank"
    for x in dev_clean; do #test_clean train_clean_100 train_clean_360 train_other_500; do
        # make log mel-scale filter bank
        sed -e "s/wav\//lmfb\//g" -e "s/\.wav/\.npy/g" data/${x}/wavlist > data/${x}/mellist
        cut -d '/' data/${x}/mellist -f -12 | sed -e "s/^/mkdir -p /g" | bash
        paste -d ' ' data/${x}/wavlist data/${x}/mellist > tmp/scp.wav2mel
        python utils/wav2lmfb.py tmp/scp.wav2mel
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Text preparation"
    for x in dev_clean test_clean train_clean_100 train_clean_360; do
        python tools/lower_word.py data/${x}/text > data/${x}/text.lower
    done
fi
