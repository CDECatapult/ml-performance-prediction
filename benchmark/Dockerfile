FROM tensorflow/tensorflow:1.10.1-gpu

COPY . /model
WORKDIR /model

ENTRYPOINT ["/bin/bash", \
            "run_bm.sh"]
