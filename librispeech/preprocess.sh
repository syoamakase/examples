#!/bin/bash

stage=-1       # start from -1 if you need to start from data download
stop_stage=100

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
# datadir=/export/a15/vpanayotov/data
datadir=/n/work2/ueno/data/librispeech/data

# base url for downloads.
data_url=www.openslr.org/resources/12

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
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
    echo "stage 1: Feature Extraction"
    mkdir -p tmp

    for x in dev_clean test_clean train_clean_100 train_clean_360; do
        cut -d ' ' -f 2 data/${x}/wav.scp > data/${x}/wavlist
        python tools/extract_subdir.py data/${x}/wavlist data/${x}/${wav_dir} -n -2 | sed -e "s/^/mkdir -p /g" | bash
        # TODO: downsample
        python tools/make_convert_flac2wav.py data/${x}/wavlist --save_dir data/${x}/${wav_dir} -n -3 > tmp/convert_flac2wav.sh
        bash tmp/convert_flac2wav.sh
        python preprocess.py --hp_file ${hparams_path} -d data/${x}/${wav_dir} --resume
    done
    mkdir -p data/${dir_name}/wav.segment
    cut -d ' ' data/${dir_name}/wav.scp -f 1 | sed -e "s/^/mkdir -p data\/${dir_name}\/wav.segment\//g" | bash
    python tools/convert_kaldi2sox.py data/${dir_name}/segments data/${dir_name}/wav.scp data/${dir_name} > tmp/convert.sh
    bash tmp/convert.sh
    # make log mel-scale filter bank
    cut -d ' ' -f 3 tmp/convert.sh > data/${dir_name}/wavlist
    sed -e "s/wav\.segment/mel/g" -e "s/\.wav/\.htk/g" data/${dir_name}/wavlist > data/${dir_name}/mellist
    cut -d ' ' data/${dir_name}/wav.scp -f 1 | sed -e "s/^/mkdir -p data\/${dir_name}\/mel\//g" | bash
    paste -d ' ' data/${dir_name}/wavlist data/${dir_name}/mellist > tmp/scp.wav2mel
    HCopy -C config/config.lmfb.40ch.static -S tmp/scp.wav2mel
fi