FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

SHELL ["/bin/bash", "-c"]

RUN apt update && apt install -y git

WORKDIR /workspace

COPY . .

RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda create -y -n ste-gan python=3.10 && \
    conda init && \
    source ~/.bashrc && \
    conda run -n ste-gan pip install -r requirements.txt && \
    conda run -n ste-gan pip install . && \
    conda run -n ste-gan python -c "\
import torch.nn as nn; \
nn.utils.parametrizations.weight_norm = nn.utils.weight_norm; \
import torch; \
torch.hub.load('bshall/hubert:main', 'hubert_soft', trust_repo=True)" && \
    conda run -n ste-gan hf download ntinosbarmpas/NeuroRVQ \
    pretrained_models/tokenizers/NeuroRVQ_EMG_tokenizer_v1.pt \
    --local-dir ./ste_gan/pretrained_models

RUN echo "conda activate ste-gan" >> ~/.bashrc

ENV PATH=/opt/conda/envs/ste-gan/bin:$PATH
