# SFT-warm-start targeted GRPO restart

Goal: continue GRPO from the final SFT checkpoint and beat both eval200 baselines.

Baselines:
- Vanilla strategy generator: 165/200
- SFT strategy generator: 171/200
- Previous GRPO run: 169/200

SFT checkpoint:

```text
tinker://7b8366b9-a2da-50c0-ad7b-ae45ebbaf067:train:0/weights/qwen36-strategy-sft-20260601T1935Z-final
```

Target set:

```text
runs/strategy_grpo/targeted_task_sets/sft_eval200_nonpass_20260611T1345Z.txt
```

Launch command:

```bash
env \
  RUN_STAMP=20260611T1349Z \
  RUN_ID=qwen36-strategy-grpo-sft-targeted-nonpass-20260611T1349Z \
  SPLIT=eval \
  TASK_IDS_FILE=runs/strategy_grpo/targeted_task_sets/sft_eval200_nonpass_20260611T1345Z.txt \
  TASK_SAMPLING=sequential \
  TASKS_PER_STEP=2 \
  MAX_STEPS=15 \
  ADVANTAGE_GROUP_SIZE=8 \
  ROLLOUT_POOL_PER_TASK=16 \
  GROUPS_PER_UPDATE=4 \
  LEARNING_RATE=5e-8 \
  EXECUTOR_WORKERS=2 \
  CODEX_PROVIDER=llmgw \
  SAVE_AFTER_UPDATE=1 \
  INIT_STATE_PATH=tinker://7b8366b9-a2da-50c0-ad7b-ae45ebbaf067:train:0/weights/qwen36-strategy-sft-20260601T1935Z-final \
  scripts/run_strategy_grpo_background.sh
```

The command does not stop the older targeted-v3 eval. It uses two executor workers to reduce contention while that eval is still active.
