
dataset=vqav2

CUDA_VISIBLE_DEVICES=0,1

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} deepspeed eval/eval_vqa.py \
        --json_path /dataset/vqa_dataset/vqa_v2/bunny_vqav2_mscoco_test-dev2015.jsonl \
        --image_folder /dataset/vqa_dataset/vqa_v2/test2015 \
        --model_path /model/HyperSeg-3B \
        --answers_file /output/vqa_v2/${CHUNKS}_${IDX}.jsonl \
        --eval_dataset vqav2 \
        --num_chunks ${CHUNKS} \
        --chunk_idx ${IDX} &
done

wait

output_file=/output/vqa_v2/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"


# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /output/vqa_v2/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python hyperseg/eval/vqa/vqav2/convert_vqav2_for_submission.py \
    --src ${output_file} \
    --dst /output/vqav2/vqav2_answers_upload.json \

