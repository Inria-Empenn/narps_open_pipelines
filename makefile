.PHONY: Dockerfile

Dockerfile:
	docker run --rm repronim/neurodocker:0.9.5 generate docker \
			--base-image centos:7 --pkg-manager yum \
			--yes \
			--install git \
			--ants method=binaries version=2.4.3 \
			--fsl version=6.0.6.4 \
			--spm12 version=r7771 method=binaries \
			--user=neuro \
			--workdir /home \
			--miniconda method=binaries env_name=neuro \
						version=latest \
						conda_install="python=3.11 traits jupyter nilearn graphviz nipype==1.8.6 scikit-image" \
						pip_install="matplotlib" \
			--env LD_LIBRARY_PATH="/opt/miniconda-latest/envs/neuro:$LD_LIBRARY_PATH" \
			--run-bash "source activate neuro" \
			--user=root \
			--run 'chmod 777 -Rf /home' \
			--run 'chown -R neuro /home' \
			--user=neuro \
			--run 'mkdir -p ~/.jupyter && echo c.NotebookApp.ip = \"0.0.0.0\" > ~/.jupyter/jupyter_notebook_config.py' > Dockerfile