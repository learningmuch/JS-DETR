**JS-DETR**: DETR with joint position channel attention and shape adaptation improvement loss
========



## Environment Preparation

```python
pip install -r requirements.txt
```



## Data preparation

Put your coco format dataset in the data directory
We expect the directory structure to be the following:

```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```



## Training

To train baseline JS-DETR on a single node with single gpus for 300 epochs run:
```requirements.txt
python main.py --coco_path /path/to/coco 
```



## Evaluation
Put the circle weights you trained in the resume directory 

To evaluate JS-DETR on COCO val5k with a single GPU run:

```python
python main.py --batch_size 2 --no_aux_loss --eval --resume /path/to.resume --coco_path /path/to/coco
```
