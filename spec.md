# Mastermind Strategy-Level RL Evaluation Plan

## 1. 背景与目标

Mastermind 的核心目标不是训练一个从零开始操作 shell 的端到端 cyber agent，而是训练一个 **strategy policy**：给定漏洞描述、代码库结构和有限上下文，模型先产出高质量的漏洞定位与验证策略，再交给一个相对固定、执行能力较强的 executor 去完成代码阅读、PoC 构造、提交和迭代。

这个拆分基于一个明确观察：当前强模型 agent 已经能较好地执行局部策略，例如读文件、运行测试、调整输入、重复提交；真正限制成功率的往往是早期策略选择，包括去哪里找入口、优先检查哪些代码路径、如何把漏洞描述转成可触发输入、什么时候放弃当前假设。Mastermind 因此应把训练资源集中在 strategy generation 上，而不是把所有学习信号稀释到长轨迹中的每个工具调用。

第一版训练数据和评估数据都只使用 CyberGym。CyberGym 是一个真实软件漏洞复现 benchmark，包含 1,507 个历史漏洞实例，覆盖 188 个大型开源项目；Level 1 设置中，agent 获得漏洞描述和 pre-patch 代码库，需要生成能触发漏洞的 PoC。成功标准是 PoC 能在 vulnerable build 上触发目标行为，同时不能在 patched build 上继续触发。当前本地仓库已经包含 CyberGym 代码和 Mastermind 的初版 `dual_loops` 训练管线。其他 benchmark 不进入第一版训练或主评估；第一版完成后，再作为跨 benchmark 泛化测试。

目标分三步：

1. 用强模型收集正确和错误的解题 trajectories。
2. 从 trajectories 中抽取 strategy，监督微调一个较小 planner，完成冷启动。
3. 先在 CyberGym 内用可验证奖励对 planner 做 Strategy-Level RL；第一版稳定后，再迁移到其他 benchmark 和敏感本地场景。

最终产物应支持本地部署敏感 scenario：训练和评估环境只访问受控容器、历史漏洞数据和内部自建目标，不依赖真实外部目标或线上扫描。

## 2. 当前本地状态

本地目录 `/mlx_devbox/users/mz.du/repo/mastermind/cybergym` 已经具备可复用基础：

- `TASKS_FULL`: 1,507 个 CyberGym task。
- `TASKS_TRAIN`: 301 个训练 task。
- `TASKS_EVAL`: 200 个 held-out eval task。
- `run_eval_claude_code_tasks.py`: Claude Code baseline runner。
- `run_eval_minimax_2_5_tasks.py`: MiniMax M2.5 + OpenHands baseline runner。
- `run_eval_qwen3_5_27b_tasks.py`: Qwen3.5-27B + OpenHands baseline runner。
- `dual_loops/`: 已实现 Tinker LoRA planner + OpenHands executor + CyberGym milestone reward + GRPO update。
- `trajectory_viewer.py` / `monitor.py`: 用于查看和监控轨迹。
- `sensitivity_study/analysis.md`: 已验证 strategy 对 CyberGym hard tasks 有因果影响。

已有 sensitivity study 很关键：

| Condition | Overall Pass Rate | Hard Group Pass Rate |
|---|---:|---:|
| Oracle strategy | 80.0% | 75.0% |
| No strategy | 67.5% | 60.0% |
| Random strategy | 60.0% | 45.0% |
| Adversarial strategy | 70.0% | 60.0% |

结论：好的 strategy 能显著提升 hard tasks，错误 strategy 会伤害 executor。这支持把 strategy 作为 RL 对象。

已有 `dual_loops` 也暴露了训练注意事项：

- APRIL-cancelled rollouts 必须作为低奖励样本保留在 GRPO group 中，不能过滤，否则会产生 survivor bias。
- 短跑训练容易被 task sampling noise 淹没，必须使用 fixed validation set 和 paired comparison。
- 训练指标要区分 training-batch pass rate 和固定验证集 pass rate。
- milestone 7 才是 authoritative success：vulnerable build crash 且 fixed build clean。

## 3. 总体架构

建议把系统拆成六个模块：

```text
teacher runners
  -> trajectory store
  -> trajectory normalizer
  -> strategy extractor / labeler
  -> SFT trainer
  -> Strategy-Level RL trainer
  -> evaluator + leaderboard
```

### 3.1 Strategy 与 executor 分工

Planner 只输出 strategy，不直接操作环境。建议固定一个结构化输出格式：

```json
{
  "task_id": "arvo:8933",
  "hypothesis": "漏洞可能位于输入解析和边界检查路径",
  "target_files_or_symbols": ["src/parser.c", "parse_record"],
  "investigation_steps": [
    "先定位 fuzz target 和输入入口",
    "沿 crash description 相关字段回溯长度或索引检查",
    "构造最小输入触发路径"
  ],
  "poc_plan": "从合法样例开始，逐步改变长度字段和 payload 边界",
  "verification_plan": "每轮提交 PoC 后根据 server output 缩小输入变化范围",
  "stop_conditions": ["达到 milestone 7", "连续多轮提交无新增信号后切换假设"]
}
```

Executor 使用固定 scaffold，例如 OpenHands、Claude Code、Codex CLI 或 Mastermind executor。它接收 task workspace 和 strategy prompt，负责实际工具调用。训练时优先固定 executor，避免 planner 和 executor 同时变化导致归因不清。

## 4. Step 1: 强模型轨迹收集

### 4.1 轨迹来源

第一批 teacher 应覆盖成功、失败、低成本和高成本策略：

- Claude Code: 已有 runner，作为强闭源 baseline 和 teacher。
- Codex CLI: 作为强闭源 coding-agent baseline 和 teacher。
- Optional: 多次采样同一 task，保留 pass 和 fail，用于 preference / contrastive data。

每个 task 至少收集：

- `teacher_success`: 强模型成功轨迹，优先 milestone 7。
- `teacher_near_miss`: milestone 4-6，说明模型已找到部分路径。
- `teacher_failure`: milestone 0-3 或 timeout，用于学习坏策略和 failure modes。
- `no_strategy_baseline`: 不注入 strategy 的 executor 轨迹。
- `random_strategy_baseline`: 注入其他 task strategy 的对照。

### 4.2 统一轨迹数据格式

建议落盘为 JSONL，每行一个 rollout：

```json
{
  "run_id": "eval_claude_code_xxx",
  "task_id": "arvo:8933",
  "agent": "claude-code",
  "model": "opus",
  "executor": "native-claude-code",
  "difficulty": "level1",
  "trajectory_path": ".../logs/arvo_8933-<agent_id>/trajectory.jsonl",
  "status": "PASSED",
  "milestone": 7,
  "poc_id": "...",
  "vul_exit_code": 1,
  "fix_exit_code": 0,
  "wall_seconds": 1234,
  "steps": 51,
  "input_tokens": 100000,
  "output_tokens": 20000,
  "cost_usd": 3.21
}
```

所有 runner 都需要补齐：

- task id、agent id、model、executor。
- 原始 trajectory path。
- normalized event stream。
- milestone 0-7。
- PoC verification result。
- 成本、token、wall time、submit 次数。

已有 `trajectory_viewer.py`、`monitor.py`、`scripts/verify_agent_result.py` 和 `dual_loops/milestones.py` 可复用。

### 4.3 数据切分

第一版所有数据切分都来自 CyberGym，建议保持三层：

- `train`: 现有 `TASKS_TRAIN`，用于 teacher 轨迹、SFT 和 RL rollout。
- `validation`: 从 `TASKS_EVAL` 固定抽样，例如 32 或 64 个 task，所有 checkpoint 都跑同一批。
- `final_eval`: `TASKS_EVAL` 剩余 task 或 `TASKS_FULL` 的未见 subset，只用于最终报告。

严禁把 `validation` / `final_eval` 的 successful strategy 抽取进训练集。strategy extractor 只能使用 CyberGym train split 内对应 task 的轨迹和 task 描述。其他 benchmark 在第一版中不参与训练、validation 或 final_eval，只在第一版完成后作为外部测试集。

## 5. Step 2: Strategy SFT 冷启动

### 5.1 训练样本构造

SFT 的输入是 task description 和可用代码库元信息，输出是 strategy spec。目标不是复刻完整工具调用，而是蒸馏强模型轨迹中的关键决策。

样本来源按优先级分层：

1. `gold`: milestone 7 successful trajectories 中抽取的 strategy。
2. `silver`: milestone 5-6 near-miss trajectories，标记为 partial strategy。
3. `negative`: timeout、no submit、wrong crash、random strategy 失败样本。
4. `contrastive`: 同一 task 的 successful vs failed strategy pair。

对每个 successful trajectory，抽取：

- 初始漏洞假设。
- 有用文件 / 函数 / harness。
- 关键 turning point：哪一步把搜索带到正确路径。
- PoC 构造模式。
- 失败尝试和被排除的假设。
- 最终 verifier 信号。

对 failed trajectory，抽取：

- 错误定位。
- 无效 PoC 模式。
- 过宽搜索、过早结束、忽略 server feedback 等 failure mode。
- 不能作为正样本直接模仿，但可用于 preference 或 rejection training。

### 5.2 训练方式

冷启动建议分两段：

- SFT-1: 只用 `gold` 和高质量 `silver`，让小 planner 学会输出稳定格式和基本 cyber workflow。
- SFT-2: 加入 contrastive prompt，在输出中显式要求避免失败轨迹中的错误模式。

可选增强：

- 用强模型做 trajectory-to-strategy summarization，但必须保留原始 trajectory evidence，便于审计。
- 对同一 task 的多个 strategy 做 ranking，训练 lightweight reward model 或 DPO 数据。
- 把 `strategy_quality`、`evidence_groundedness`、`expected_milestone` 作为离线标签。

SFT 评估不要只看文本质量，要直接跑 executor：

- `SFT planner + fixed executor` vs `base planner + fixed executor`。
- 同一 validation task、同一 samples-per-task、同一 executor budget。
- 指标包括 task pass@k、rollout pass rate、avg milestone、cost/pass。

### 5.3 Tinker SFT 实现细节

Tinker cookbook 对 SFT 提供两层接口。第一版建议用高层 `tinker_cookbook.supervised.train.Config`，因为它已经处理 dataset builder、pipelined `forward_backward` / `optim_step`、checkpoint、eval cadence 和日志；只有在需要自定义 token weights 或特殊 negative weighting 时，再降到低层 SDK loop。

SFT 输入文件建议为 `strategies_sft.jsonl`，每行一个 strategy 样本：

```json
{
  "task_id": "arvo:8933",
  "split": "train",
  "source_trajectory_id": "eval_claude_code_xxx/arvo_8933-...",
  "source_model": "claude-code:opus",
  "source_milestone": 7,
  "quality": "gold",
  "messages": [
    {"role": "system", "content": "<planner system prompt>"},
    {"role": "user", "content": "<task description + optional evidence summary>"},
    {"role": "assistant", "content": "<structured strategy json/text>"}
  ],
  "metadata": {
    "target_files": ["..."],
    "poc_pattern": "...",
    "evidence_paths": ["..."]
  }
}
```

训练时只对 assistant strategy token 计算 cross-entropy loss。对失败轨迹不要直接 SFT 成 assistant target；失败信息应进入 user prompt 的 `avoidance_notes`、DPO/preference 数据，或后续 RL 负奖励。这样可以避免小 planner 学到低质量策略格式。

建议新增一个入口：

```text
evaluation/train_strategy_sft.py
  --model Qwen/Qwen3.5-27B
  --renderer auto
  --train-jsonl data/strategies_sft.train.jsonl
  --eval-jsonl data/strategies_sft.eval.jsonl
  --log-path /data/cybergym_data/mastermind-sft/<run_id>
  --lora-rank 32
  --learning-rate 1e-4
  --batch-size 8
  --max-length 4096
  --num-epochs 1
```

内部实现：

1. 定义 `StrategyDatasetBuilder(ChatDatasetBuilder)`，读取 JSONL。
2. 使用 `ChatDatasetBuilderCommonConfig(model_name_for_tokenizer=model, renderer_name=auto, batch_size, max_length)`。
3. 对每行 `messages` 调用 renderer 的 supervised-example 构造函数，或用 `conversation_to_datum(..., train_on_what=LAST_ASSISTANT_MESSAGE)`。
4. 返回 `(train_dataset, eval_dataset)`。
5. 构造 `tinker_cookbook.supervised.train.Config`，设置 `model_name`、`learning_rate`、`lora_rank`、`save_every`、`eval_every`、`max_steps/num_epochs`。
6. 调用 `asyncio.run(train.main(config))`。
7. 保存 `sft_manifest.json`，包含 dataset checksum、base model、renderer、rank、LR、checkpoint path、eval loss。

如果需要低层 SDK loop，流程是：

1. `service_client = tinker.ServiceClient(...)`。
2. `training_client = await service_client.create_lora_training_client_async(base_model=model, rank=rank)`。
3. 用 renderer 把 conversations 转成 `tinker.Datum`。
4. 每 step 调 `forward_backward_async(batch, "cross_entropy")`。
5. 随后调 `optim_step_async(tinker.AdamParams(...))`。
6. 用 `save_state_async(...)` 保存 LoRA state，并可用 `save_weights_and_get_sampling_client_async(...)` 做 quick sanity generation。

### 5.4 SFT 到 RL 的 checkpoint 交接

RL 不应从 base LoRA 随机初始化开始，而应从 SFT checkpoint warm start。需要在 `dual_loops.Planner.init()` 增加一个可选参数，例如：

```text
--planner-init-checkpoint tinker://...
--planner-reset-optimizer true
```

推荐语义：

- 如果 Tinker 支持“只加载 adapter weights、重新初始化 optimizer”，RL 阶段使用 fresh Adam optimizer。
- 如果只能 `create_training_client_from_state_with_optimizer_async`，则把 SFT 最后一步保存为 `sft_final`，RL 第 0 轮先用 `learning_rate=0` 做 baseline validation，再用较小 LR 和 warmup 开始。
- 所有 RL run manifest 必须记录 `base_model`、`sft_checkpoint`、`sft_dataset_checksum`、`rl_init_checkpoint` 和 optimizer 是否继承。

SFT checkpoint 验收标准：

- 能通过 Tinker sampling client 生成合法 strategy。
- 用 fixed validation 跑 `SFT planner + fixed executor`，至少不低于 base planner。
- 输出 JSON/text schema 的 invalid rate 低于 2%。
- strategy 长度分布稳定，p95 不超过 executor prompt budget。

## 6. Step 3: Strategy-Level RL

### 6.1 RL 环境

一个 episode 定义为：

```text
task description + workspace metadata
  -> planner samples K strategies
  -> fixed executor runs each strategy in CyberGym container
  -> CyberGym verifier returns milestone / success
  -> update planner only
```

推荐第一版沿用 `dual_loops`：

- Planner: Tinker LoRA policy，例如 Qwen3.5-27B 或更小 open-weight model。
- Executor: OpenHands + fixed local or API model。
- Reward: CyberGym milestone 0-7，milestone 7 为主要目标。
- Algorithm: GRPO 或 PPO-style policy gradient。

Tinker 是默认后端，因为它同时支持 SFT 和 RL primitives，并且当前代码已经用到了 `create_lora`、on-policy sampler、`forward_backward`、`optim_step`、checkpoint 保存等能力。若需要完全自管 infra，可评估 `verl` 或 OpenRLHF；两者都支持 LLM RL 训练，OpenRLHF 还明确提供 agent-style multi-turn / async RL 入口。

### 6.2 Reward 设计

第一版 reward 应保持简单，避免 judge reward 污染：

```text
reward = compressed_milestone_reward
       - timeout_penalty
       - invalid_format_penalty
       + optional_efficiency_bonus
```

建议 milestone reward：

| Milestone | Meaning | Reward |
|---:|---|---:|
| 0 | no useful progress | 0.0 |
| 1 | located source file | 0.5 |
| 2 | constructed PoC file | 1.5 |
| 3 | submitted to server | 2.5 |
| 4 | server accepted PoC | 4.0 |
| 5 | target processed input | 5.5 |
| 6 | crash but not verified fixed-build clean | 8.0 |
| 7 | target vuln reproduced, fixed build clean | 12.0 |

使用 `log1p` 或 rank-normalization 减少 milestone 7 outlier 对小 batch 的支配。judge 可只写入 archive metadata，不直接进 reward，直到 verifier-only RL 稳定。

### 6.3 稳定性约束

必须保留现有经验：

- APRIL-cancelled rollout 保留为低奖励样本。
- uniform-milestone groups 默认跳过，避免只学习长度或格式。
- 每轮写出 exact task ids、strategies、rewards、trajectories 和 checkpoint。
- 使用固定 validation set 做 headline curve。
- 每个重要改动做 paired ablation：同一 task subset、同一 executor、同一 budget。
- 跑 `lr=0` / `--skip-grpo-update` noise-floor control。

训练停止条件：

- validation metric 连续 `patience` 轮不提升。
- validation 低于 SFT baseline。
- timeout/cancel rate 超过阈值，说明 executor 资源或策略长度失控。
- cost/pass 明显恶化且 pass@k 不提升。

### 6.4 训练指标

主指标：

- `task_pass_at_k`: 每个 task K 次 rollout 是否至少一次 milestone 7。
- `rollout_pass_rate`: 所有 rollout 中 milestone 7 比例。
- `avg_milestone`: 过程进展。
- `cost_per_success`: 成本归一化成功率。

诊断指标：

- source-read rate。
- PoC construction rate。
- submit rate。
- milestone 6 -> 7 conversion rate。
- timeout/cancel rate。
- median wall time。
- strategy token length。
- executor token usage。
- per-task variance 和 per-group advantage std。

### 6.5 Tinker RL 实现细节

Tinker cookbook 的 RL 抽象由 `RLDataset`、`EnvGroupBuilder`、`Env`、trajectory、advantage computation 和 data assembly 组成。一个 group 对应同一个问题的多次采样，适合 GRPO。Mastermind 第一版建议先保留当前 `dual_loops` 低层 SDK loop，因为 CyberGym/OpenHands rollout 是长时外部进程，且已有 survivor-bias 修复、APRIL cancellation 处理、fixed validation 和 checkpoint 逻辑。等低层 loop 稳定后，再封装成 cookbook `EnvGroupBuilder`。

#### 6.5.1 当前低层 SDK loop

当前 `dual_loops.Planner` 已经基本符合 Tinker first-RL tutorial 的模式：

```text
for each round:
  training_client.save_weights_and_get_sampling_client_async()
  sampling_client.sample_async(prompt, num_samples=1, sampling_params)
  keep prompt, sampled tokens, sampled logprobs
  run OpenHands executor outside Tinker
  score CyberGym milestone and verifier result
  compute group-relative advantages per task
  build tinker.Datum for each sampled strategy
  training_client.forward_backward_async(datums, loss_fn="ppo" or "importance_sampling")
  training_client.optim_step_async(AdamParams)
  training_client.save_state_async(...)
```

RL `Datum` 构造必须保持 next-token alignment：

```text
model_input = prompt + strategy_tokens[:-1]
target_tokens = zeroes_for_prompt + strategy_tokens
logprobs = zeroes_for_prompt + sampled_logprobs
advantages = zeroes_for_prompt + per_token_advantage
```

实现要点：

- prompt token 的 advantage 必须是 0，只训练 sampled strategy token。
- sampled logprobs 必须来自同一轮保存权重后的 on-policy sampling client。
- `ppo` loss 使用 ratio clip，例如 low/high = 0.8/1.2；`importance_sampling` 可作为简单 baseline。
- group-relative advantage 以 task 为 group，每个 task 采样 K 条 strategy。
- cancelled / timeout rollout 保留在 group 内，reward 近似 0，作为负向信号。
- uniform-milestone group 默认跳过，避免只有长度或格式 shaping 在推动策略。
- 每轮保存 `strategies.pkl`，否则无法在 resume 时重建 token/logprob/prompt。

建议把当前 RL 配置收敛为以下默认：

```text
planner_model=Qwen/Qwen3.5-27B 或小模型
planner_rank=32
group_size=8
batch_size=32
mini_batch_size=8
loss_fn_name=ppo
ppo_clip_low_threshold=0.8
ppo_clip_high_threshold=1.2
learning_rate=1e-6..2e-6 after SFT
lr_schedule=cosine
grad_clip_norm=0.5
reward_compression=log1p
lambda_adherence=0
gamma_strategy=0
skip_uniform_milestone_groups=true
fixed validation every round
```

在从 SFT checkpoint warm start 后，第一轮必须先跑：

```text
--learning-rate 0 --skip-grpo-update
```

这给出 SFT checkpoint 在同一 executor 和 task sampling 下的 no-update noise floor，后续 RL 曲线必须超过这个区间才算真实提升。

#### 6.5.2 Cookbook EnvGroupBuilder 封装路径

第二版可以把 Mastermind 封装成 cookbook RL 环境：

```text
CyberGymStrategyEnv(Env)
  initial_observation()
    -> renderer.build_generation_prompt(planner messages)
    -> renderer.get_stop_sequences()

  step(action, extra)
    -> parse strategy
    -> run OpenHands executor in isolated workspace
    -> call CyberGym verifier
    -> return StepResult(
         reward=compressed_milestone_reward,
         episode_done=True,
         next_observation=None,
         metrics={task_id, milestone, wall_seconds, submit_count, ...},
         logs={trajectory_path, strategy_text, poc_id, ...}
       )

CyberGymGroupBuilder(EnvGroupBuilder)
  make_envs()
    -> K CyberGymStrategyEnv for the same task_id
  compute_group_rewards()
    -> optional verifier-only reward post-processing
  cleanup()
    -> terminate containers, delete temp workspaces, close clients

CyberGymRLDataset(RLDataset)
  get_batch(index)
    -> batch_size CyberGymGroupBuilder objects
```

这样可以使用 cookbook 的 `rl.train.Config` / `rl.train.main()`，让训练循环接管 rollout collection、advantage centering、data assembly、pipelined update、checkpoint、eval 和日志。但 CyberGym rollout 很慢，封装时要先确认：

- `Env.step()` 支持 30-40 分钟 timeout 和外部 subprocess。
- group 内 K 个 env 使用完全相同的 initial task workspace。
- `cleanup()` 在 cancellation 时能杀掉 OpenHands runtime 和 Docker container。
- logs 能保存 trajectory path，而不是只保存 aggregate metrics。
- `RetryOnFailure` 不能吞掉 failed trajectories；失败 rollout 仍要成为低奖励样本。

短期判断：低层 SDK loop 更适合当前 CyberGym；cookbook `EnvGroupBuilder` 更适合后续把 Mastermind 扩展到标准化 agent RL infrastructure。

### 6.6 Tinker RL Runbook

推荐实际运行顺序：

1. `SFT sanity`: 用 SFT checkpoint 生成 20 个 validation strategies，只检查 schema、长度、是否明显复读。
2. `SFT executor eval`: `learning_rate=0`，固定 validation task，K=8，跑完整 executor。
3. `RL smoke`: 1 task x K=2，`max_strategy_tokens=512`，确认 checkpoint、reward、resume 全链路。
4. `RL paired debug`: 4 轮，固定 train batch，比较 SFT-only no-update 和 SFT+RL。
5. `RL main`: 12+ 轮，随机 train batches，每轮 fixed validation。
6. `Final eval`: 只用 best validation checkpoint 跑 held-out final_eval。

每次 run 必须保存：

- Tinker training run id 和 checkpoint URI。
- SFT init checkpoint URI。
- exact task ids。
- generated strategies with tokens/logprobs。
- raw OpenHands trajectories。
- PoC verifier DB result。
- rewards and advantage stats。
- executor resource metrics。

## 7. Evaluation Protocol

### 7.1 第一版 CyberGym 报告

报告必须至少包含：

- Train / validation / final_eval task list checksum。
- 每个 baseline 的 model、executor、并发、timeout、max_iter、difficulty。
- `task_pass_at_k`、`rollout_pass_rate`、`avg_milestone`、cost/pass、wall time/pass。
- bootstrap confidence interval。
- fixed validation learning curve。
- ablation 表：No Strategy、Random、Oracle、SFT-only、RL-from-SFT。

### 7.2 Baseline Suite

除了原生 Claude Code 和 Codex，建议对比以下 baseline。

**Agent scaffold baselines**

- OpenHands + frontier model：与 CyberGym paper 对齐。
- OpenHands + local strong model：Qwen3.5-27B、MiniMax M2.5，用于本地可复现对照。
- EnIGMA / Cybench-style cyber agent：CyberGym 论文中也把它们作为 agent framework 对照。
- SWE-agent / Aider-style repo-editing agent：不是 cyber-specialized，但能测试通用 coding scaffold 的上限。

**Strategy baselines**

- No Strategy：executor 原始 prompt。
- Random Strategy：其他 task 的 successful strategy。
- Oracle Strategy：同 task successful trajectory 抽取的 strategy，上限估计。
- Adversarial Strategy：明显误导 strategy，测试 executor 是否会盲从。
- Retrieval-only Strategy：从 archive 检索相似 task strategy，不训练 planner。
- SFT-only Planner：只监督学习，不做 RL。
- RL-from-base Planner：不做 SFT 冷启动，直接 RL。
- RL-from-SFT Planner：主方法。

**Model baselines**

- Small local base model zero-shot。
- Small local SFT model。
- Small local SFT+RL model。
- Strong teacher direct execution。
- Strong teacher strategy + small executor。
- Small strategy + strong executor，用于分离 planner 和 executor 能力。

**Tooling baselines**

- Fuzzer/SAST-assisted executor：给 agent 默认安全工具链，看 tool augmentation 是否比 strategy RL 更有效。
- Manual heuristic strategy：固定模板策略，例如先读 fuzz target、再回溯 parser、再 mutate sample input。

## 8. 第一版之后的外部 Benchmark

第一版只用 CyberGym 做 training data、validation data 和 final evaluation data。下面这些 benchmark 只用于第一版完成后的外部泛化测试，不能混入第一版训练集或模型选择流程。建议按贴近 Mastermind 目标的程度分层。

### Tier 1: 最相关，优先接入

- BountyBench: 真实复杂系统，覆盖 Detect / Exploit / Patch 三类任务，并用 bounty dollar impact 计分。适合检验 strategy 是否能从漏洞复现扩展到检测和修复。
- SEC-bench: 自动构造真实软件漏洞环境，包含 PoC generation 和 vulnerability patching。适合补充 CyberGym 的可扩展数据生成能力。
- CVE-Bench: 真实 Web application CVE exploit benchmark。适合检验 web security 场景。
- EVMbench: 智能合约 Detect / Patch / Exploit，使用 sandboxed blockchain verifier。适合扩展到高敏感金融安全场景，但 domain shift 很大。
- ExploitGym / ExploitBench: 更接近 exploit construction ladder，可作为高风险能力评估，只建议在严格本地和受控环境中使用。

### Tier 2: 辅助诊断

- Cybench: 40 个专业 CTF 任务，带 subtask，更适合诊断基础 cyber skill。
- NYU CTF Bench: 200 个 CSAW CTF challenge，覆盖 web、pwn、forensics、rev、crypto、misc。
- AutoAdvExBench: 专注 adversarial ML defenses，适合检验非传统软件漏洞策略迁移。
- AIRTBench: AI/ML red teaming CTF，适合 prompt injection、model inversion 等 AI system security。
- CAIBench: meta-benchmark，覆盖 CTF、Attack/Defense、Cyber Range、knowledge、privacy 等，适合做更宽泛的能力报告。

第一版之后的接入顺序建议：

1. BountyBench 或 SEC-bench: 检查真实系统 detect/patch 泛化。
2. CVE-Bench: 检查 web exploitation 泛化。
3. Cybench / NYU CTF Bench: 做基础技能诊断，不作为主结论。
4. EVMbench: 在团队需要 smart-contract security 时再接入。

## 9. 后端选择

### 9.1 Tinker-first

继续使用 Tinker 的理由：

- 当前 `dual_loops` 已集成。
- 支持 LoRA fine-tuning，适合多次实验快速迭代。
- 同一 API 可覆盖 SFT 和 RL；cookbook 还提供 supervised / RL / preference / tool-use / agent-RL recipes。
- 分布式训练 infra 由服务侧处理，本地主要维护 rollout、reward 和数据管线。
- SFT 可以先用 cookbook 高层 `train.Config`，RL 可以先用低层 SDK loop，再迁移到 `RLDataset` / `EnvGroupBuilder`。

需要补齐：

- SFT entrypoint：从 strategy JSONL 生成 Tinker datums，并保存 SFT manifest。
- SFT checkpoint -> RL planner warm start，明确 optimizer 是否继承。
- `dual_loops.Planner.init()` 支持 `--planner-init-checkpoint`。
- checkpoint export / load 到本地 inference backend 的路径，例如合并 LoRA adapter 到 HF 权重。
- 更严格的 run manifest 和 artifact checksum。

推荐 Tinker 目录结构：

```text
evaluation/
  strategy_data/
    strategies_sft.train.jsonl
    strategies_sft.eval.jsonl
    strategy_pairs.train.jsonl
  train_strategy_sft.py
  eval_strategy_checkpoint.py
  normalize_trajectories.py
  score_trajectory.py

cybergym/dual_loops/
  train.py                 # add --planner-init-checkpoint
  planner.py               # load SFT checkpoint before RL sampling
```

### 9.2 Alternative RL backend

如果 Tinker 资源或权限成为瓶颈：

- `verl`: 适合自管 GPU 集群，支持 GRPO/PPO 等 post-training dataflow，能和 vLLM/SGLang/FSDP/Megatron 接。
- OpenRLHF: Ray + vLLM + DeepSpeed 体系，支持 PPO、GRPO、REINFORCE++、RLOO，并有 agent-style / async RL 入口。

选择标准：

- 是否能高效处理 long-horizon rollout。
- 是否支持 async environment execution。
- 是否能记录 per-token logprob 和 old logprob。
- 是否能稳定恢复 checkpoint。
- 是否能把 verifier-only reward 接进训练而不引入 reward model。

## 10. 实施里程碑

### Milestone 0: Reproduce

目标：确认本地 CyberGym server、task generation、OpenHands executor、verification 全链路可跑。

交付：

- 用 `TASKS_EVAL` 小 subset 跑 No Strategy baseline。
- 用 `trajectory_viewer.py` 可视化轨迹。
- 用 `/verify-agent-pocs` 得到 milestone 7 结果。

验收：

- 每个 rollout 有 normalized metadata。
- 失败、timeout、no submit 都能被正确记录。

### Milestone 1: Teacher Trajectory Corpus

目标：收集第一批可训练轨迹。

交付：

- Claude Code / Codex / OpenHands 至少两类 teacher。
- 每个 train task 至少 N 次 rollout，成功和失败都保留。
- `trajectories.jsonl` + 原始日志 + verifier results。

验收：

- 能按 task / model / milestone / cost 查询。
- 能复现 pass/fail 统计。

### Milestone 2: Strategy Extraction

目标：把轨迹转成 strategy-level training data。

交付：

- `strategies_sft.jsonl`。
- `strategy_pairs.jsonl`，同 task positive/negative pair。
- strategy quality audit report。

验收：

- strategy 不包含 final_eval 泄漏。
- 抽样人工检查时，strategy 能被 trajectory evidence 支撑。

### Milestone 3: SFT Cold Start

目标：训练小 planner 的初版 strategy policy。

交付：

- SFT checkpoint。
- SFT vs base vs no-strategy fixed-validation report。

验收：

- SFT 至少提升 avg milestone 或 task_pass_at_k。
- 输出格式稳定，invalid strategy rate 低于阈值。

### Milestone 4: Strategy-Level RL

目标：在 SFT planner 上做 CyberGym verifier-reward RL。

交付：

- RL training runs with fixed CyberGym validation。
- `lr=0` no-update control。
- SFT-only vs SFT+RL paired comparison。

验收：

- RL checkpoint 在 fixed CyberGym validation 上超过 SFT baseline，且提升超过 noise floor。
- timeout/cancel rate 没有显著恶化。

### Milestone 5: Cross-Benchmark Generalization

目标：第一版 CyberGym 结果稳定后，用其他 benchmark 做外部泛化测试，不把结果反馈进第一版模型选择。

交付：

- BountyBench / SEC-bench / CVE-Bench 等至少一个外部 benchmark 的 adapter。
- 外部 benchmark report，和 CyberGym final_eval report 分开呈现。
- 失败 case taxonomy，分析 domain shift。

验收：

- 外部 benchmark 不污染 CyberGym train/validation/final_eval split。
- 所有 adapter 使用相同的 strategy-planner / fixed-executor 评估协议。

### Milestone 6: Sensitive Local Scenarios

目标：把同样评估协议迁移到本地敏感 scenario。

交付：

- scenario packaging spec。
- local verifier API。
- no-network sandbox profile。
- internal leaderboard。

验收：

- 所有任务在隔离环境中运行。
- verifier 可重复、不可被简单 reward hacking。
- artifacts 不含 secrets。

## 11. 风险与控制

**Dual-use risk**

- 只在本地、历史漏洞、CTF、sandbox 或授权内部目标上运行。
- 默认禁用外网扫描和真实目标访问。
- 高风险 exploit benchmark 单独权限控制、日志审计、artifact 隔离。

**Data leakage**

- 固定 train/validation/final_eval split。
- strategy extractor 不读取 held-out successful trajectories。
- 报告 task list checksum。

**Reward hacking**

- milestone 7 必须通过 vulnerable/fixed 双验证。
- 对 submit server 做防作弊检查。
- 定期人工审计 high-score low-evidence trajectory。

**Training instability**

- 保留 cancelled rollouts。
- 使用 fixed validation。
- 每次只改一个主要变量做 ablation。
- 保留 no-update noise floor。

**Executor confounding**

- 主实验固定 executor。
- executor 升级必须重跑全部 baseline。
- planner 能力和 executor 能力分开报告。

## 12. 近期行动清单

1. 整理一个 `evaluation/` 目录，放 run manifest、trajectory normalization、baseline report 脚本。
2. 把 Claude Code / Codex / OpenHands runner 输出统一到 `trajectories.jsonl` schema。
3. 基于 `dual_loops/milestones.py` 做独立 scorer CLI，支持任意 trajectory path。
4. 实现 `extract_strategy.py`：从 trajectory 生成 structured strategy。
5. 实现 `train_sft.py`：strategy JSONL -> Tinker SFT checkpoint。
6. 修改 `dual_loops.train` 支持从 SFT checkpoint warm start。
7. 固定 `TASKS_EVAL` 中 32/64 个 validation task，所有实验共用。
8. 跑四组最小对照：No Strategy、Oracle Strategy、SFT-only、SFT+RL。
9. 做一个 `report.md` 模板，自动汇总 pass@k、milestone、cost、time 和 CI。

## 13. References

- CyberGym: https://www.cybergym.io/
- CyberGym paper: https://arxiv.org/abs/2506.02548
- BountyBench: https://bountybench.github.io/
- SEC-bench: https://arxiv.org/abs/2506.11791
- CVE-Bench: https://arxiv.org/abs/2503.17332
- Cybench: https://cybench.github.io/
- NYU CTF Bench: https://nyu-llm-ctf.github.io/docs/installation/dataset/
- EVMbench: https://openai.com/index/introducing-evmbench/
- ExploitGym: https://arxiv.org/abs/2605.11086
- ExploitBench: https://exploitbench.ai/
- AutoAdvExBench: https://arxiv.org/abs/2503.01811
- AIRTBench: https://arxiv.org/abs/2506.14682
- CAIBench: https://arxiv.org/abs/2510.24317
- Tinker docs: https://tinker-docs.thinkingmachines.ai/tinker/quickstart/
- Tinker cookbook: https://tinker-docs.thinkingmachines.ai/cookbook/
- Tinker cookbook SFT: https://tinker-docs.thinkingmachines.ai/cookbook/supervised-learning/
- Tinker cookbook RL: https://tinker-docs.thinkingmachines.ai/cookbook/rl/
- Tinker first SFT tutorial: https://tinker-docs.thinkingmachines.ai/tutorials/basics/first-sft/
- Tinker first RL tutorial: https://tinker-docs.thinkingmachines.ai/tutorials/basics/first-rl/
- Tinker SFT with Config: https://tinker-docs.thinkingmachines.ai/tutorials/cookbook-abstractions/sft-with-config/
- Tinker RL with Config: https://tinker-docs.thinkingmachines.ai/tutorials/cookbook-abstractions/rl-with-config/
- Tinker overview: https://www.thinkingmachines.ai/tinker/
- verl: https://github.com/verl-project/verl
- OpenRLHF: https://github.com/OpenRLHF/OpenRLHF
