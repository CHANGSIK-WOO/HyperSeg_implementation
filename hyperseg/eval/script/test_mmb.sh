


CUDA_VISIBLE_DEVICES=0,1

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} deepspeed eval/eval_vqa_mmb.py \
        --json_path /dataset/vqa_dataset/mmb/MMBench_DEV_EN_legacy.tsv \
        --model_path /model/HyperSeg-3B \
        --answers_file /output/mmb/${CHUNKS}_${IDX}.jsonl \
        --eval_dataset mmb \
        --num_chunks ${CHUNKS} \
        --chunk_idx ${IDX} &
done

wait

output_file=/output/mmb/merge.jsonl
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /output/mmb/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

mkdir -p /output/mmb/answers_upload

python hyperseg/eval/vqa/mmbench/convert_mmbench_for_submission.py \
    --annotation-file /dataset/vqa_dataset/mmb/MMBench_DEV_EN_legacy.tsv \
    --result-dir /output/mmb/merge.jsonl \
    --upload-dir /output/mmb/answers_upload \
    --experiment mmb_filter \