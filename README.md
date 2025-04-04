# Cubismo

![cubismo logo](assets/cubismo.jpg)

Cubismo is a space where I explore image generation by training a Flow Matching model to generate human faces.

## How it works

Relies on `pytorch` and Meta's [`flow_matching` library](https://github.com/facebookresearch/flow_matching)


## How to use

```bash
git clone https://github.com/GrmanRodriguez/cubismo.git
cd cubismo
pip install -e .
python train/train.py
```