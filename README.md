# Cubismo

![cubismo logo](assets/cubismo.jpg)

Cubismo is a space where I explore image generation by training a Flow Matching model to generate human faces.

The images are not in cubist style (at least, not on purpose 😛).

## How it works

Relies on `pytorch` and Meta's [`flow_matching` library](https://github.com/facebookresearch/flow_matching)

For data, I'm using the [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset.


## How to use

```bash
git clone https://github.com/GrmanRodriguez/cubismo.git
cd cubismo
pip install -e .
python scripts/train.py
python scripts/try_model.py --checkpoint /path/to/checkpoint
```