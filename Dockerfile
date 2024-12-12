FROM nipype/nipype:py38

WORKDIR /work

COPY narps_open/ ./narps_open/
COPY setup.py ./

RUN /neurodocker/startup.sh pip install .
