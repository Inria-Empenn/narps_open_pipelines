FROM nipype/nipype:py38
COPY . /work
USER root
RUN /bin/bash -c "source activate neuro && pip install /work"
USER neuro
