conda create -n MLLM-code python=3.10.14
conda activate MLLM-code
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt