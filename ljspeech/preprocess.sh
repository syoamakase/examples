#!/bin/bash

stage=-1
stop_stage=100
# For alignment for Fastspeech 2
export CUDA_VISIBLE_DEVICES=0

sampling_rate=16000
wav_dir='wav_16kHz'
mode='Transformer' # or FastSpeech 2 Transformer
script_dir=script_${sampling_rate}

echo ${script_dir}

mkdir -p data/train data/dev
mkdir -p data/train/${wav_dir}
mkdir -p data/dev/${wav_dir}
mkdir -p data/train/mean_var_${sampling_rate}
mkdir -p data/train/${script_dir}
mkdir -p data/dev/${script_dir}
mkdir -p tmp

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Data prepartion"
    readlink -f LJSpeech-1.1/wavs/* > wavlist  
    python tools/read_labfiles.py LJSpeech-1.1/metadata.csv > data/train/train_all.txt
    head -n 50 data/train/train_all.txt > data/dev/dev.txt
    head -n 50 wavlist > data/dev/wavlist
    tail -n+50 data/train/train_all.txt > data/train/train.txt
    tail -n+50 wavlist > data/train/wavlist
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Feature Generation"
    if [ ${sampling_rate} = "22050" ]; then
        cat data/train/wavlist | sed -e "s/^/cp /g" -e "s/$/ data\/train\/${wav_dir}/g" | bash
        cat data/dev/wavlist | sed -e "s/^/cp /g" -e "s/$/ data\/dev\/${wav_dir}/g" | bash
    else
        python tools/make_sox.py data/train/wavlist --save_dir data/train/${wav_dir} --sampling_rate ${sampling_rate} | bash
        python tools/make_sox.py data/dev/wavlist --save_dir data/dev/${wav_dir} --sampling_rate ${sampling_rate} | bash
    fi
    if [ ${mode} = "FastSpeech2" ]; then
        python preprocess.py --hp_file configs/hparams_${sampling_rate}.py -d data/train/${wav_dir} --fastspeech2
        python preprocess.py --hp_file configs/hparams_${sampling_rate}.py -d data/dev/${wav_dir} --fastspeech2
    else
        python preprocess.py --hp_file configs/hparams_${sampling_rate}.py -d data/train/${wav_dir}
        python preprocess.py --hp_file configs/hparams_${sampling_rate}.py -d data/dev/${wav_dir}
    fi
fi

if  [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Text prepartion"
    python tools/add_fullpath.py data/train/train.txt --save_dir data/train/${wav_dir} --ext npy > data/train/${script_dir}/train_id.txt
    python tools/add_fullpath.py data/dev/dev.txt --save_dir data/dev/${wav_dir} --ext npy > data/dev/${script_dir}/dev_id.txt

    python tools/sort_by_xlen.py -S data/train/${script_dir}/train_id.txt > data/train/${script_dir}/train_id_sort_xlen.txt
    python tools/feature_normalize.py -S data/train/${script_dir}/train_id.txt --ext npy --save_dir data/train/${script_dir}/
fi

if  [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && [ ${mode} = "FastSpeech2" ]; then
    echo "Get the alignment from it"
    if [ ${sampling_rate} = "22050" ]; then
        python tools/alignment/get_alignment.py --load_name tools/alignment/pretrained_models/checkpoints.LJSpeech_22.05kHz_ctc.phones/network.epoch25 --test_script data/train/${script_dir}/train_id_sort_xlen.txt
    else
        python tools/alignment/get_alignment.py --load_name tools/alignment/pretrained_models/checkpoints.LJSpeech_16kHz_ctc.phones/network.epoch25 --test_script data/train/${script_dir}/train_id_sort_xlen.txt
    fi
fi
