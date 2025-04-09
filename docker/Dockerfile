FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ARG CHECKPOINT_PATH

COPY . /cubismo

RUN pip install /cubismo

COPY ${CHECKPOINT_PATH} /checkpoint.ckpt

WORKDIR /cubismo/server

CMD ["uvicorn", "app:app"]