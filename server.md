```shell
$ export greedy='false'
$ export top_p=0.8
$ export top_k=20
$ export temperature=0.7
$ export repetition_penalty=1.0
$ export presence_penalty=1.5
$ export out_seq_length=16384
$ vllm serve ./model/Qwen/Qwen3-VL-4B-Instruct \
  --max_model_len 1024 \
  --enable-multimodal \
  --max_num_batched_tokens 1024 \
  --tensor-parallel-size 1 \
  --mm-encoder-tp-mode data \
  --async-scheduling \
  --host 0.0.0.0 \
  --port 8000

```