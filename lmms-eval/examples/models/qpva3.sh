# Run and exactly reproduce qwen2vl results!
# mme as an example
export HF_HOME="~/.cache/huggingface"
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
# pip3 install qwen_vl_utils
# use `interleave_visuals=True` to control the visual token position, currently only for mmmu_val and mmmu_pro (and potentially for other interleaved image-text tasks), please do not use it unless you are sure about the operation details.

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qpva3 \
    --model_args.planner_path=Qwen/Qwen2.5-VL-7B-Instruct \
    --model_args.max_pixels=12845056 \
    --model_args.attn_implementation=flash_attention_2 \
    --model_args.interleave_visuals=False \
    --tasks agqa_decomp \
    --batch_size 1