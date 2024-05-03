# seq len 2048
export TPU_NAME='tpuvm-v1'
export ZONE='us-central2-b'


# Log per 128 * 50 steps, matching the gradient accumulation steps = Real 1 step
gcloud compute tpus tpu-vm ssh beomi@$TPU_NAME --zone $ZONE --worker=all --command "
export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_enable_async_all_gather=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'

python -m EasyLM.models.gemma.gemma_train \
--load_checkpoint=flax_params::mnt_ds/gemma-7B/flax_model.msgpack \
--mesh_dim=1,-1,4 \
--dtype=bf16 \
--total_steps=1000 \
--log_freq=128 \
--save_model_freq=999320000 \
--save_milestone_freq=10000 \
--train_dataset.type='huggingface' \
--train_dataset.text_processor.fields='text' \
--train_dataset.json_dataset.seq_length=8192 \
--train_dataset.json_dataset.batch_size=8 \
--train_dataset.json_dataset.path=mnt_ds/cached_ds \
--optimizer.accumulate_gradient_steps=64 \
--optimizer.type=adamw \
--optimizer.adamw_optimizer.weight_decay=0.1 \
--optimizer.adamw_optimizer.lr=5e-5 \
--optimizer.adamw_optimizer.end_lr=4e-5 \
--optimizer.adamw_optimizer.lr_warmup_steps=10000 \
--optimizer.adamw_optimizer.lr_decay_steps=320000 \
--checkpointer.save_optimizer_state=False \
--checkpointer.float_dtype=bf16 \
--logger.online=True \
--logger.output_dir=/gemma-checkpoint"
