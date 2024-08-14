<div align="center">

# CUSTrack: A Causal View on Ultrasound Motion Tracking with Historical Trajectory

The official implementation for the paper: <br/> [_CUSTrack: A Causal View on Ultrasound Motion Tracking
 with Historical Trajectory_]().



<a href="https://shkzhang.github.io/CUSTrack/"><img alt="Home" src="https://img.shields.io/badge/Home-CUSTrack-blue?logo=github&logoColor=ffffff"></a>
<a href="https://huggingface.co/shkzhang/CUSTrack"><img alt="model"  src="https://img.shields.io/badge/Model-CUSTrack-FFD21E?logo=huggingface&logoColor=#FFD21E&labelColor=FFD21E"></a>
<a href="https://wandb.ai/shukang/tracking/reports/CUSTrack-Training-Report--Vmlldzo4OTYyNTI5?accessToken=t0x46zti2tnutcfchkqftjl9u45sqhtmnqczbtjyzjdt4thqqsx6ty9yvicbms39"><img alt="Logs"  src="https://img.shields.io/badge/Wandb-Training%20Logs-blue?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBzdGFuZGFsb25lPSJubyI/PjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIgImh0dHA6Ly93d3cudzMub3JnL0dyYXBoaWNzL1NWRy8xLjEvRFREL3N2ZzExLmR0ZCI+PHN2ZyB0PSIxNzIyNTEwNzk4ODY3IiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9Ijg1NjgiIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCI+PHBhdGggZD0iTTAgMzMwLjc1MmMwIDU4LjQ0OTkyIDQ3LjM4MDQ4IDEwNS44NDA2NCAxMDUuODQwNjQgMTA1Ljg0MDY0IDU4LjQ0OTkyIDAgMTA1Ljg0MDY0LTQ3LjM5MDcyIDEwNS44NDA2NC0xMDUuODQwNjRzLTQ3LjM5MDcyLTEwNS44NDA2NC0xMDUuODQwNjQtMTA1Ljg0MDY0QzQ3LjM4MDQ4IDIyNC45MTEzNiAwIDI3Mi4zMDIwOCAwIDMzMC43NTJ6TTAgODU5Ljk1NTJjMCA1OC40Mzk2OCA0Ny4zODA0OCAxMDUuODMwNCAxMDUuODQwNjQgMTA1LjgzMDQgNTguNDQ5OTIgMCAxMDUuODQwNjQtNDcuMzgwNDggMTA1Ljg0MDY0LTEwNS44NDA2NCAwLTU4LjQ0OTkyLTQ3LjM5MDcyLTEwNS44NDA2NC0xMDUuODQwNjQtMTA1Ljg0MDY0QzQ3LjM4MDQ4IDc1NC4xMDQzMiAwIDgwMS40OTUwNCAwIDg1OS45NTUyek0zOS42OTAyNCA1OTUuMzUzNmE2Ni4xNTA0IDY2LjE1MDQgMCAxIDAgMTMyLjMwMDggMCA2Ni4xNTA0IDY2LjE1MDQgMCAwIDAtMTMyLjMwMDggMHpNMzkuNjkwMjQgNjYuMTUwNGE2Ni4xNTA0IDY2LjE1MDQgMCAxIDAgMTMyLjMwMDggMCA2Ni4xNTA0IDY2LjE1MDQgMCAwIDAtMTMyLjMwMDggMHpNNDA2LjE1OTM2IDY5My4yNDhjMCA1OC40NDk5MiA0Ny4zODA0OCAxMDUuODQwNjQgMTA1Ljg0MDY0IDEwNS44NDA2NCA1OC40NDk5MiAwIDEwNS44NDA2NC00Ny4zOTA3MiAxMDUuODQwNjQtMTA1Ljg0MDY0UzU3MC40NDk5MiA1ODcuNDA3MzYgNTEyIDU4Ny40MDczNmMtNTguNDYwMTYgMC0xMDUuODQwNjQgNDcuMzkwNzItMTA1Ljg0MDY0IDEwNS44NDA2NHpNNDQ1Ljg0OTYgOTU3Ljg0OTZhNjYuMTUwNCA2Ni4xNTA0IDAgMSAwIDEzMi4zMDA4IDAgNjYuMTUwNCA2Ni4xNTA0IDAgMCAwLTEzMi4zMDA4IDB6TTQ0NS44NDk2IDQyOC42NDY0YTY2LjE1MDQgNjYuMTUwNCAwIDEgMCAxMzIuMzAwOCAwIDY2LjE1MDQgNjYuMTUwNCAwIDAgMC0xMzIuMzAwOCAwek00NDUuODQ5NiAxNjQuMDQ0OGE2Ni4xNTA0IDY2LjE1MDQgMCAxIDAgMTMyLjMwMDggMCA2Ni4xNTA0IDY2LjE1MDQgMCAwIDAtMTMyLjMwMDggMHpNODEyLjMxODcyIDMzMC43NTJjMCA1OC40NDk5MiA0Ny4zOTA3MiAxMDUuODQwNjQgMTA1Ljg0MDY0IDEwNS44NDA2NEM5NzYuNjE5NTIgNDM2LjU5MjY0IDEwMjQgMzg5LjIwMTkyIDEwMjQgMzMwLjc1MnMtNDcuMzgwNDgtMTA1Ljg0MDY0LTEwNS44NDA2NC0xMDUuODQwNjRjLTU4LjQ0OTkyIDAtMTA1Ljg0MDY0IDQ3LjM5MDcyLTEwNS44NDA2NCAxMDUuODQwNjR6TTg1Mi4wMDg5NiA2Ni4xNTA0YTY2LjE1MDQgNjYuMTUwNCAwIDEgMCAxMzIuMzAwOCAwIDY2LjE1MDQgNjYuMTUwNCAwIDAgMC0xMzIuMzAwOCAwek04NTIuMDA4OTYgNTk1LjM1MzZhNjYuMTUwNCA2Ni4xNTA0IDAgMSAwIDEzMi4zMDA4IDAgNjYuMTUwNCA2Ni4xNTA0IDAgMCAwLTEzMi4zMDA4IDB6TTg1Mi4wMDg5NiA4NTkuOTU1MmE2Ni4xNTA0IDY2LjE1MDQgMCAxIDAgMTMyLjMwMDggMCA2Ni4xNTA0IDY2LjE1MDQgMCAwIDAtMTMyLjMwMDggMHoiIHAtaWQ9Ijg1NjkiIGZpbGw9IiNmZmZmZmYiPjwvcGF0aD48L3N2Zz4="></a>
<a href="https://github.com/shkzhang/CUSTrack?tab=MIT-1-ov-file"><img alt="License" src="https://img.shields.io/badge/License-MIT-green?logoColor=ffffff&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBzdGFuZGFsb25lPSJubyI/PjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIgImh0dHA6Ly93d3cudzMub3JnL0dyYXBoaWNzL1NWRy8xLjEvRFREL3N2ZzExLmR0ZCI+PHN2ZyB0PSIxNzIyNTEwNzAxMTQwIiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9Ijc0ODkiIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCI+PHBhdGggZD0iTTMwOS43NjgwODkgMTk1LjYyNjQ2OWMwLTEyLjYzNDc2MiAxMC4yMzcxNTUtMjIuODgxMTI3IDIyLjg2NzgyNC0yMi44ODExMjcgMTIuNjMxNjkyIDAgMjIuODU4NjE0IDEwLjI0NjM2NSAyMi44NTg2MTQgMjIuODgxMTI3IDAgMTIuNjM4ODU1LTEwLjIyNzk0NSAyMi44NzQ5ODctMjIuODU4NjE0IDIyLjg3NDk4N0MzMjAuMDEyNDA4IDIxOC41MDY1NzMgMzA5Ljc2ODA4OSAyMDguMjY1MzI0IDMwOS43NjgwODkgMTk1LjYyNjQ2OUwzMDkuNzY4MDg5IDE5NS42MjY0Njl6TTIxNS43MTQ5OTMgMTk1LjYyNjQ2OWMwLTEyLjYzNDc2MiAxMC4yMzkyMDItMjIuODgxMTI3IDIyLjg2OTg3LTIyLjg4MTEyNyAxMi42MjQ1MjkgMCAyMi44NTc1OTEgMTAuMjQ2MzY1IDIyLjg1NzU5MSAyMi44ODExMjcgMCAxMi42Mzg4NTUtMTAuMjMzMDYyIDIyLjg3NDk4Ny0yMi44NTc1OTEgMjIuODc0OTg3QzIyNS45NTQxOTUgMjE4LjUwNjU3MyAyMTUuNzE0OTkzIDIwOC4yNjUzMjQgMjE1LjcxNDk5MyAxOTUuNjI2NDY5TDIxNS43MTQ5OTMgMTk1LjYyNjQ2OXpNNjUwLjk0MzQ5MiA3MjkuODk4NzI5IDM2Mi4wOTg5NDUgNzI5Ljg5ODcyOWMtMjkuMzYyNzQ4IDAtMjkuMzYyNzQ4LTM3LjY3NTA2NCAwLTM3LjY3NTA2NGwyODguODQ0NTQ3IDBDNjc2LjE0NjUwMSA2OTIuMjEzNDMxIDY4Ny4yNDExODYgNzI5Ljg5ODcyOSA2NTAuOTQzNDkyIDcyOS44OTg3Mjl6TTY1MC45NDM0OTIgNTI4Ljk2MzM0NyAzNjIuMDk4OTQ1IDUyOC45NjMzNDdjLTI1LjczMjA1OCAwLTI3LjQ5ODI4NC0zNy42NzUwNjQgMC0zNy42NzUwNjRsMjg4Ljg0NDU0NyAwQzY4Mi4zMDI3MTEgNDkxLjI4ODI4MyA2ODUuMzkwMDI2IDUyOC4wODAyMzQgNjUwLjk0MzQ5MiA1MjguOTYzMzQ3ek0xMzYuMDQ2NTEyIDkxOC4yNzYwOTdjLTcuODk4OTAxIDAtMjUuMTE3MDUxLTE3LjIxODE1LTI1LjExNzA1MS0yNS4xMTcwNTFMMTEwLjkyOTQ2MiAzMTUuNDY4OTI4bDgwMy43NDE1MjcgMCAwIDU3Ny42ODkwOTVjMCA4LjM5ODI3NC0xNi43MTg3NzcgMjUuMTE3MDUxLTI1LjExNzA1MSAyNS4xMTcwNTFMMTM2LjA0NjUxMiA5MTguMjc1MDc0ek0xNDguNjA1NTQ5IDEwMS45NzU1MzNjMTguMjI4MTUzIDAgNzI4LjM5MTM5OSAwIDcyOC4zOTEzOTkgMCA4LjM5ODI3NCAwIDM3LjY3NTA2NCAxNi43MTM2NiAzNy42NzUwNjQgMjUuMTE3MDUxbDAgMTUwLjcwMTI4MUwxMTAuOTI5NDYyIDI3Ny43OTM4NjQgMTEwLjkyOTQ2MiAxMzkuNjUwNTk3QzEwOS43NjQ5MzkgMTEzLjk5NzMzNCAxMzAuMzc2MzczIDEwMS45NzU1MzMgMTQ4LjYwNTU0OSAxMDEuOTc1NTMzek04ODUuMjYxMTY5IDY0LjMwMTQ5MiAxNDAuMzQwMzA1IDY0LjMwMTQ5MmMtMzYuOTg5NDQ5IDAtNjcuMDg1OTA4IDMwLjExNzk0OC02Ny4wODU5MDggNjcuMTI4ODg3bDAgNzU3LjM4OTg1YzAgMzcuMDA5OTE1IDMwLjA5NjQ1OSA2Ny4xMjk5MSA2Ny4wODU5MDggNjcuMTI5OTFsNzQ0LjkxOTg0IDBjMzYuOTg5NDQ5IDAgNjcuMDg1OTA4LTMwLjExOTk5NSA2Ny4wODU5MDgtNjcuMTI5OTFsMC03NTcuMzg5ODVDOTUyLjM0NzA3NyA5NC40MTk0NCA5MjIuMjUwNjE4IDY0LjMwMTQ5MiA4ODUuMjYxMTY5IDY0LjMwMTQ5Mkw4ODUuMjYxMTY5IDY0LjMwMTQ5MnpNODg1LjI2MTE2OSA2NC4zMDE0OTIiIGZpbGw9IiNmZmZmZmYiIHAtaWQ9Ijc0OTAiPjwvcGF0aD48L3N2Zz4="></a>



<p align="center">
  <img width="85%" src="assets/frame.png" alt="Framework"/>
</p>

</div>

 **Abstract:**
 Real-time tissue tracking is a foundation task in liver ultra
sound applications. Due to the periodic nature of liver mo
tion, historical trajectories often serve as essential visual cues
 for target localization, especially in cases of poor foreground
background distinction. However, trackers with trajectory
 prompts tend to depend onthe motionperiodicity excessively,
 leading to a shortcut for motion estimation. In such cases,
 target positions are inferred primarily from recurring major
 shifts. In this paper, we revisit liver tracking with a causal
 view and propose a causal inference framework, called CUS
Track. It addresses this issue by keeping the “good” trajectory
 prior while removing the “bad” trajectory bias. Specifically,
 this method identifies trajectory bias as the direct causal ef
fect of historical states on target position, and removes it from
 the total causal effect which includes both appearance and
 trajectory information. By virtue of counterfactual reasoning
 during inference, our CUSTrack method achieves new state
of-the-arts on two liver ultrasound tracking datasets.






## Contents
- [Setup](#Setup)
  	- [Data preparation](#Data-preparation)
  	- [Install the environment](#Install-the-environment)
  	- [Set project paths](#Set-project-paths)
- [Training](#Training)
	- [Download pre-trained model](#Download-pre-trained-model)
   	- [Start training](#Start-training)
- [Evaluation](#Evaluation)
	- [Prepare model weights](#Prepare-model-weights)
	- [Testing](#Testing)
	- [Quantitative Evaluation](#Quantitative-Evaluation)
	- [Qualitative Evaluation](#Qualitative-Evaluation)
	- [Visualization](#Visualization)
- [Acknowledgments](#Acknowledgments)
- [Citation](#Citation)


<details open>

<summary ><h2 style="display: inline">Setup</h2></summary>

### Data preparation
Put the tracking datasets in ***${DATA_DIR}***. It should look like this:
   ```
   ${DATA_DIR}
     -- clust
         |-- TestSet
         |-- TrainingSet
     -- ndth
         |-- 1
         |-- 2
         |-- ...
         |-- n
   ```
### Install the environment

**Method1**: Use the Anaconda ***CUDA 11.8***
```shell
conda env create -f custrack_cuda118_env.yaml
```

**Method2**: Use the pip ***CUDA 11.8***
```shell
pip install -r requirements_cuda118.txt
```


### Set project paths
Run the following command to set paths for this project
```shell
python script/create_default_local_file.py --workspace_dir . --data_dir ${DATA_DIR} --save_dir ./output
```

</details>






<details open>

<summary ><h2 style="display: inline">Training</h2></summary>

### Download pre-trained model

Download pre-trained [MAE ViT-Base weights](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and put it under `${PROJECT_ROOT}/pretrained_models` (different pretrained models can also be used, see [MAE](https://github.com/facebookresearch/mae) for more details).

### Start training

```shell
python tracking/train.py --script custrack --config base --save_dir ./output --mode multiple --nproc_per_node 4
```

You can run the command ```python tracking/train.py --help``` to see the optional run parameters

</details>





<details open>

<summary ><h2 style="display: inline">Evaluation</h2></summary>

### Prepare model weights
Download the model weights from [Hugging Face](https://huggingface.co/shkzhang/CUSTrack) 

Put the downloaded weights on `${CHECKPOINT_PATH}` such as: `${PROJECT_ROOT}/output/checkpoints/train/custrack/base`

### Testing

- CLUST-test
```shell
python tracking/test.py custrack base --dataset clust --checkpoint ${CHECKPOINT_PATH}
```
- NDTH-test
```shell 
python tracking/test.py ostrack base --dataset ndth --checkpoint ${CHECKPOINT_PATH}
```

- Multiple dataset
```shell 
python tracking/test.py ostrack base --dataset clust,ndth --checkpoint ${CHECKPOINT_PATH}
```

The result will be saved in `${PROJECT_ROOT}/output//test/tracking_results/custrack/base_0`

### Quantitative Evaluation

```shell
python tracking/analysis_results.py 
```

### Qualitative Evaluation

```shell
python tracking/figure_result.py 
```

### Visualization
[Visdom](https://github.com/fossasia/visdom) is used for visualization. 
1. Alive visdom in the server by running `visdom`:
```shell
 python -m visdom.server
```
2. Simply set `--debug 1` during inference for visualization, e.g.:
```
python tracking/test.py custrack base --dataset clust --threads 1 --num_gpus 1 --debug 1
```
3. Open `http://localhost:8097` in your browser.



</details>

## Acknowledgments
We would like to express our gratitude to the following repositories for their contribution to our work:

- [STARK](https://github.com/researchmm/Stark)
- [PyTracking](https://github.com/visionml/pytracking)
- [Timm](https://github.com/rwightman/pytorch-image-models)
- [OSTrack](https://github.com/botaoye/OSTrack)

We acknowledge the creators and contributors of these repositories for their valuable work and open-source contributions.
## Citation
If our work is useful for your research, please consider citing:

```Bibtex

```
