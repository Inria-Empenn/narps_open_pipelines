# Generated by Neurodocker and Reproenv.

FROM centos:7
RUN yum install -y -q \
                  git \
           && yum clean all \
           && rm -rf /var/cache/yum/*
ENV ANTSPATH="/opt/ants-2.4.3/" \
    PATH="/opt/ants-2.4.3:$PATH"
RUN yum install -y -q \
           curl \
           unzip \
    && yum clean all \
    && rm -rf /var/cache/yum/* \
    && echo "Downloading ANTs ..." \
    && curl -fsSL -o ants.zip https://github.com/ANTsX/ANTs/releases/download/v2.4.3/ants-2.4.3-centos7-X64-gcc.zip \
    && unzip ants.zip -d /opt \
    && mv /opt/ants-2.4.3/bin/* /opt/ants-2.4.3 \
    && rm ants.zip
ENV FSLDIR="/opt/fsl-6.0.6.4" \
    PATH="/opt/fsl-6.0.6.4/bin:$PATH" \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLMULTIFILEQUIT="TRUE" \
    FSLTCLSH="/opt/fsl-6.0.6.4/bin/fsltclsh" \
    FSLWISH="/opt/fsl-6.0.6.4/bin/fslwish" \
    FSLLOCKDIR="" \
    FSLMACHINELIST="" \
    FSLREMOTECALL="" \
    FSLGECUDAQ="cuda.q"
RUN yum install -y -q \
           bc \
           curl \
           file \
           libGL \
           libGLU \
           libICE \
           libSM \
           libX11 \
           libXcursor \
           libXext \
           libXft \
           libXinerama \
           libXrandr \
           libXt \
           libgomp \
           libjpeg \
           libmng \
           libpng12 \
           nano \
           openblas-serial \
           python3 \
           sudo \
           wget \
    && yum clean all \
    && rm -rf /var/cache/yum/* \
    && echo "Installing FSL ..." \
    && curl -fsSL https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py | python3 - -d /opt/fsl-6.0.6.4 -V 6.0.6.4
ENV FORCE_SPMMCR="1" \
    SPM_HTML_BROWSER="0" \
    SPMMCRCMD="/opt/spm12-r7771/run_spm12.sh /opt/matlab-compiler-runtime-2010a/v713 script" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/opt/matlab-compiler-runtime-2010a/v713/runtime/glnxa64:/opt/matlab-compiler-runtime-2010a/v713/bin/glnxa64:/opt/matlab-compiler-runtime-2010a/v713/sys/os/glnxa64:/opt/matlab-compiler-runtime-2010a/v713/extern/bin/glnxa64" \
    MATLABCMD="/opt/matlab-compiler-runtime-2010a/v713/toolbox/matlab"
RUN export TMPDIR="$(mktemp -d)" \
    && yum install -y -q \
           bc \
           curl \
           libXext \
           libXmu \
           libXpm \
           libXt \
           unzip \
    && yum clean all \
    && rm -rf /var/cache/yum/* \
    && echo "Downloading MATLAB Compiler Runtime ..." \
    && curl -fL -o "$TMPDIR/MCRInstaller.bin" https://dl.dropbox.com/s/zz6me0c3v4yq5fd/MCR_R2010a_glnxa64_installer.bin \
    && chmod +x "$TMPDIR/MCRInstaller.bin" \
    && "$TMPDIR/MCRInstaller.bin" -silent -P installLocation="/opt/matlab-compiler-runtime-2010a" \
    && rm -rf "$TMPDIR" \
    && unset TMPDIR \
    # Install spm12
    && echo "Downloading standalone SPM12 ..." \
    && curl -fL -o /tmp/spm12.zip https://www.fil.ion.ucl.ac.uk/spm/download/restricted/utopia/previous/spm12_r7771_R2010a.zip \
    && unzip -q /tmp/spm12.zip -d /tmp \
    && mkdir -p /opt/spm12-r7771 \
    && mv /tmp/spm12/* /opt/spm12-r7771/ \
    && chmod -R 777 /opt/spm12-r7771 \
    && rm -rf /tmp/spm* \
    # Test
    && /opt/spm12-r7771/run_spm12.sh /opt/matlab-compiler-runtime-2010a/v713 quit
ENV CONDA_DIR="/opt/miniconda-latest" \
    PATH="/opt/miniconda-latest/bin:$PATH"
RUN yum install -y -q \
           bzip2 \
           curl \
    && yum clean all \
    && rm -rf /var/cache/yum/* \
    # Install dependencies.
    && export PATH="/opt/miniconda-latest/bin:$PATH" \
    && echo "Downloading Miniconda installer ..." \
    && conda_installer="/tmp/miniconda.sh" \
    && curl -fsSL -o "$conda_installer" https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash "$conda_installer" -b -p /opt/miniconda-latest \
    && rm -f "$conda_installer" \
    && conda update -yq -nbase conda \
    # Prefer packages in conda-forge
    && conda config --system --prepend channels conda-forge \
    # Packages in lower-priority channels not considered if a package with the same
    # name exists in a higher priority channel. Can dramatically speed up installations.
    # Conda recommends this as a default
    # https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-channels.html
    && conda config --set channel_priority strict \
    && conda config --system --set auto_update_conda false \
    && conda config --system --set show_channel_urls true \
    # Enable `conda activate`
    && conda init bash \
    && conda install -y  --name base \
           "python=3.11" \
           "pip=23.2.1" \
    && bash -c "source activate base \
    &&   python -m pip install --no-cache-dir  \
             "traits==6.3.0" \
             "jupyterlab-4.0.6" \
             "graphviz-0.20.1" \
             "nipype==1.8.6" \
             "scikit-image==0.21.0" \
             "matplotlib==3.8.0" \
             "nilearn==0.10.1"" \
    # Clean up
    && sync && conda clean --all --yes && sync \
    && rm -rf ~/.cache/pip/*
RUN mkdir -p ~/.jupyter && echo c.NotebookApp.ip = \"0.0.0.0\" > ~/.jupyter/jupyter_notebook_config.py

# Save specification to JSON.
RUN printf '{ \
  "pkg_manager": "yum", \
  "existing_users": [ \
    "root" \
  ], \
  "instructions": [ \
    { \
      "name": "from_", \
      "kwds": { \
        "base_image": "centos:7" \
      } \
    }, \
    { \
      "name": "install", \
      "kwds": { \
        "pkgs": [ \
          "git" \
        ], \
        "opts": null \
      } \
    }, \
    { \
      "name": "run", \
      "kwds": { \
        "command": "yum install -y -q \\\\\\n           git \\\\\\n    && yum clean all \\\\\\n    && rm -rf /var/cache/yum/*" \
      } \
    }, \
    { \
      "name": "env", \
      "kwds": { \
        "ANTSPATH": "/opt/ants-2.4.3/", \
        "PATH": "/opt/ants-2.4.3:$PATH" \
      } \
    }, \
    { \
      "name": "run", \
      "kwds": { \
        "command": "yum install -y -q \\\\\\n    curl \\\\\\n    unzip\\nyum clean all\\nrm -rf /var/cache/yum/*\\necho \\"Downloading ANTs ...\\"\\ncurl -fsSL -o ants.zip https://github.com/ANTsX/ANTs/releases/download/v2.4.3/ants-2.4.3-centos7-X64-gcc.zip\\nunzip ants.zip -d /opt\\nmv /opt/ants-2.4.3/bin/* /opt/ants-2.4.3\\nrm ants.zip" \
      } \
    }, \
    { \
      "name": "env", \
      "kwds": { \
        "FSLDIR": "/opt/fsl-6.0.6.4", \
        "PATH": "/opt/fsl-6.0.6.4/bin:$PATH", \
        "FSLOUTPUTTYPE": "NIFTI_GZ", \
        "FSLMULTIFILEQUIT": "TRUE", \
        "FSLTCLSH": "/opt/fsl-6.0.6.4/bin/fsltclsh", \
        "FSLWISH": "/opt/fsl-6.0.6.4/bin/fslwish", \
        "FSLLOCKDIR": "", \
        "FSLMACHINELIST": "", \
        "FSLREMOTECALL": "", \
        "FSLGECUDAQ": "cuda.q" \
      } \
    }, \
    { \
      "name": "run", \
      "kwds": { \
        "command": "yum install -y -q \\\\\\n    bc \\\\\\n    curl \\\\\\n    file \\\\\\n    libGL \\\\\\n    libGLU \\\\\\n    libICE \\\\\\n    libSM \\\\\\n    libX11 \\\\\\n    libXcursor \\\\\\n    libXext \\\\\\n    libXft \\\\\\n    libXinerama \\\\\\n    libXrandr \\\\\\n    libXt \\\\\\n    libgomp \\\\\\n    libjpeg \\\\\\n    libmng \\\\\\n    libpng12 \\\\\\n    nano \\\\\\n    openblas-serial \\\\\\n    python3 \\\\\\n    sudo \\\\\\n    wget\\nyum clean all\\nrm -rf /var/cache/yum/*\\n\\necho \\"Installing FSL ...\\"\\ncurl -fsSL https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py | python3 - -d /opt/fsl-6.0.6.4 -V 6.0.6.4" \
      } \
    }, \
    { \
      "name": "env", \
      "kwds": { \
        "FORCE_SPMMCR": "1", \
        "SPM_HTML_BROWSER": "0", \
        "SPMMCRCMD": "/opt/spm12-r7771/run_spm12.sh /opt/matlab-compiler-runtime-2010a/v713 script", \
        "LD_LIBRARY_PATH": "$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/opt/matlab-compiler-runtime-2010a/v713/runtime/glnxa64:/opt/matlab-compiler-runtime-2010a/v713/bin/glnxa64:/opt/matlab-compiler-runtime-2010a/v713/sys/os/glnxa64:/opt/matlab-compiler-runtime-2010a/v713/extern/bin/glnxa64", \
        "MATLABCMD": "/opt/matlab-compiler-runtime-2010a/v713/toolbox/matlab" \
      } \
    }, \
    { \
      "name": "run", \
      "kwds": { \
        "command": "export TMPDIR=\\"$\(mktemp -d\)\\"\\nyum install -y -q \\\\\\n    bc \\\\\\n    curl \\\\\\n    libXext \\\\\\n    libXmu \\\\\\n    libXpm \\\\\\n    libXt \\\\\\n    unzip\\nyum clean all\\nrm -rf /var/cache/yum/*\\necho \\"Downloading MATLAB Compiler Runtime ...\\"\\ncurl -fL -o \\"$TMPDIR/MCRInstaller.bin\\" https://dl.dropbox.com/s/zz6me0c3v4yq5fd/MCR_R2010a_glnxa64_installer.bin\\nchmod +x \\"$TMPDIR/MCRInstaller.bin\\"\\n\\"$TMPDIR/MCRInstaller.bin\\" -silent -P installLocation=\\"/opt/matlab-compiler-runtime-2010a\\"\\nrm -rf \\"$TMPDIR\\"\\nunset TMPDIR\\n# Install spm12\\necho \\"Downloading standalone SPM12 ...\\"\\ncurl -fL -o /tmp/spm12.zip https://www.fil.ion.ucl.ac.uk/spm/download/restricted/utopia/previous/spm12_r7771_R2010a.zip\\nunzip -q /tmp/spm12.zip -d /tmp\\nmkdir -p /opt/spm12-r7771\\nmv /tmp/spm12/* /opt/spm12-r7771/\\nchmod -R 777 /opt/spm12-r7771\\nrm -rf /tmp/spm*\\n# Test\\n/opt/spm12-r7771/run_spm12.sh /opt/matlab-compiler-runtime-2010a/v713 quit" \
      } \
    }, \
    { \
      "name": "env", \
      "kwds": { \
        "CONDA_DIR": "/opt/miniconda-latest", \
        "PATH": "/opt/miniconda-latest/bin:$PATH" \
      } \
    }, \
    { \
      "name": "run", \
      "kwds": { \
        "command": "yum install -y -q \\\\\\n    bzip2 \\\\\\n    curl\\nyum clean all\\nrm -rf /var/cache/yum/*\\n# Install dependencies.\\nexport PATH=\\"/opt/miniconda-latest/bin:$PATH\\"\\necho \\"Downloading Miniconda installer ...\\"\\nconda_installer=\\"/tmp/miniconda.sh\\"\\ncurl -fsSL -o \\"$conda_installer\\" https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh\\nbash \\"$conda_installer\\" -b -p /opt/miniconda-latest\\nrm -f \\"$conda_installer\\"\\nconda update -yq -nbase conda\\n# Prefer packages in conda-forge\\nconda config --system --prepend channels conda-forge\\n# Packages in lower-priority channels not considered if a package with the same\\n# name exists in a higher priority channel. Can dramatically speed up installations.\\n# Conda recommends this as a default\\n# https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-channels.html\\nconda config --set channel_priority strict\\nconda config --system --set auto_update_conda false\\nconda config --system --set show_channel_urls true\\n# Enable `conda activate`\\nconda init bash\\nconda install -y  --name base \\\\\\n    \\"python=3.11\\" \\\\\\n    \\"pip=23.2.1\\"\\nbash -c \\"source activate base\\n  python -m pip install --no-cache-dir  \\\\\\n      \\"traits==6.3.0\\" \\\\\\n      \\"jupyterlab-4.0.6\\" \\\\\\n      \\"graphviz-0.20.1\\" \\\\\\n      \\"nipype==1.8.6\\" \\\\\\n      \\"scikit-image==0.21.0\\" \\\\\\n      \\"matplotlib==3.8.0\\" \\\\\\n      \\"nilearn==0.10.1\\"\\"\\n# Clean up\\nsync && conda clean --all --yes && sync\\nrm -rf ~/.cache/pip/*" \
      } \
    }, \
    { \
      "name": "run", \
      "kwds": { \
        "command": "mkdir -p ~/.jupyter && echo c.NotebookApp.ip = \\\\\\"0.0.0.0\\\\\\" > ~/.jupyter/jupyter_notebook_config.py" \
      } \
    } \
  ] \
}' > /.reproenv.json
# End saving to specification to JSON.
