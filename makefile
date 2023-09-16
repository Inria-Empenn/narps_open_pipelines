.PHONY: Dockerfile

Dockerfile:
	docker run --rm repronim/neurodocker:0.9.5 generate docker \
			--base-image centos:7 --pkg-manager yum \
			--yes \
			--install git \
			--ants method=binaries version=2.4.3 \
			--fsl version=6.0.6.4 \
			--spm12 version=r7771 method=binaries \
			--miniconda method=binaries \
						version=latest 
						conda_install="python=3.10 pip=23.2.1" \
						pip_install="traits==6.3.0 jupyterlab-4.0.6 graphviz-0.20.1 nipype==1.8.6 scikit-image==0.21.0 matplotlib==3.8.0 nilearn==0.10.1" \
			--run 'mkdir -p ~/.jupyter && echo c.NotebookApp.ip = \"0.0.0.0\" > ~/.jupyter/jupyter_notebook_config.py' > Dockerfile

Dockerfile_mamba:
	neurodocker generate docker \
			--base-image centos:7 --pkg-manager yum \
			--yes \
			--install git \
			--ants method=binaries version=2.4.3 \
			--fsl version=6.0.6.4 \
			--spm12 version=r7771 method=binaries \
			--miniconda method=binaries \
						version=latest \
						mamba=true \
						conda_install="python=3.10 pip=23.2.1" \
						pip_install="traits==6.3.0 jupyterlab-4.0.6 graphviz-0.20.1 nipype==1.8.6 scikit-image==0.21.0 matplotlib==3.8.0 nilearn==0.10.1" \
			--run 'mkdir -p ~/.jupyter && echo c.NotebookApp.ip = \"0.0.0.0\" > ~/.jupyter/jupyter_notebook_config.py' > Dockerfile