# Set up the environment to run pipelines

## Run a docker container :whale:

Start a container using the command below:

```bash
docker run -ti \
		-p 8888:8888 \
		elodiegermani/open_pipeline
```

On this command line, you need to add volumes to be able to link with your local files (original dataset and git repository). If you stored the original dataset in `data/original`, just make a volume with the `narps_open_pipelines` directory:

```bash
docker run 	-ti \
		-p 8888:8888 \
		-v /users/egermani/Documents/narps_open_pipelines:/home/ \
		elodiegermani/open_pipeline
``` 

If it is in another directory, make a second volume with the path to your dataset:

```bash
docker run 	-ti \
		-p 8888:8888 \
		-v /Users/egermani/Documents/narps_open_pipelines:/home/ \
		-v /Users/egermani/Documents/data/NARPS/:/data/ \
		elodiegermani/open_pipeline
```

After that, your container will be launched! 

## Other useful docker commands

### START A CONTAINER 

```bash
docker start [name_of_the_container]
```

### VERIFY A CONTAINER IS IN THE LIST 

```bash
docker ps
```

### EXECUTE BASH OR ATTACH YOUR CONTAINER 

```bash
docker exec -ti [name_of_the_container] bash
```

**OR**

```bash
docker attach [name_of_the_container]
```

## Useful commands inside the container

### ACTIVATE CONDA ENVIRONMENT

```bash
source activate neuro
```

### LAUNCH JUPYTER NOTEBOOK

```bash
jupyter notebook --port=8888 --no-browser --ip=0.0.0.0
```

## If you did not use your container for a while

Verify it still runs :

```bash
docker ps -l
```

If your container is in the list, run :

```bash
docker start [name_of_the_container]
```

Else, relaunch it with : 

```bash
docker run 	-ti \
		-p 8888:8888 \
		-v /home/egermani:/home \
		[name_of_the_image]
```

### To use SPM inside the container, use this command at the beginning of your script:

```python
from nipype.interfaces import spm

matlab_cmd = '/opt/spm12-r7771/run_spm12.sh /opt/matlabmcr-2010a/v713/ script'

spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)
```
