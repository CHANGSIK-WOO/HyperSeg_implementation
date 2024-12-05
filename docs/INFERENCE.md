# Running Inference with HyperSeg


We provide the pretrained HyperSeg-3B model weights. Please download them from [HyperSeg-3B](https://huggingface.co/weic22/HyperSeg-3B) and put them under the current path.

## RES (RefCOCO/+/g)
```
deepspeed /eval/eval_refcoco.py \
  --image_folder /dataset/coco/train2014 \
  --json_path /dataset/RES/refcoco/refcoco_val.json \
  --model_path /model/HyperSeg-3B \
  --output_dir /output/RES \
```


## ReasonSeg
```
deepspeed /eval/eval_ReasonSeg.py \
  --reason_path /dataset/ReasonSeg \
  --model_path /model/HyperSeg-3B \
  --output_dir /output/ReasonSeg \
  --reason_seg_data ReasonSeg|val \
```


## ReasonVOS
```
deepspeed /eval/eval_ReasonVOS.py \
  --revos_path /dataset/ReVOS \
  --model_path /model/HyperSeg-3B \
  --save_path /output/ReasonVOS \
```

