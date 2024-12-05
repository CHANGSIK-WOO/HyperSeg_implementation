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


## MMBench

Refer to [MMBench GitHub](https://github.com/open-compass/MMBench) to download the benchmark dataset.

```shell
sh hyperseg/eval/script/test_mmb.sh
```

The response file can be found in `/output/mmb/answers_upload`. You can submit the Excel file to [submission link](https://mmbench.opencompass.org.cn/mmbench-submission) to obtain the evaluation scores.



## VQAv2

Refer to [here](https://github.com/BAAI-DCAI/Bunny/blob/main/script/eval/full/evaluation_full.md#vqav2) to prepare the VQAv2 benchmark dataset.

```shell
sh hyperseg/eval/script/test_vqav2.sh
```

The response file can be found in `/output/vqav2/vqav2_answers_upload.json`. You can submit the `json` response file to [submission link](https://eval.ai/web/challenges/challenge-page/830) (Test-Dev Phase) to obtain the evaluation scores.