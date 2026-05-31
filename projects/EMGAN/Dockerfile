FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

SHELL ["/bin/bash", "-c"]

RUN apt update && apt install -y git

RUN cd dgm-2026.1/projects/EMGAN && \
    source /opt/conda/etc/profile.d/conda.sh && \
    conda create -y -n ste-gan python=3.10 && \
    conda init && \
    source ~/.bashrc && \
    conda run -n ste-gan pip install -r requirements.txt && \
    conda run -n ste-gan pip install -e . && \
    conda run -n ste-gan python -c "\
import torch.nn as nn; \
nn.utils.parametrizations.weight_norm = nn.utils.weight_norm; \
import torch; \
torch.hub.load('bshall/hubert:main', 'hubert_soft', trust_repo=True)"

RUN echo "conda activate ste-gan" >> ~/.bashrc

ENV PATH=/opt/conda/envs/ste-gan/bin:$PATH
