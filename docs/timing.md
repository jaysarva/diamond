Timing breakdown

Wall clock per epoch:
- `timing/epoch_wall_sec` is measured in `Trainer.run` using `time.time()` and represents end-to-end epoch duration.

Breakdown keys (all `timing/*_sec`):
- `env_interaction`: time spent stepping real environments. Timed in `src/coroutines/env_loop.py` around `env.step(...)` when the env is a `TorchEnv` (CPU-dominated).
- `imagination_rollout`: time spent in world-model rollouts excluding diffusion sampling. Timed in `src/envs/world_model_env.py` around the non-diffusion portions of `WorldModelEnv.step(...)` (GPU-dominated, synchronized).
- `diffusion_sampling_teacher`/`diffusion_sampling_student`: time spent inside diffusion sampling for the world model. Timed in `src/envs/world_model_env.py` around `predict_next_obs()` with the role label (GPU-dominated, synchronized).
- `policy_value_update`: time spent training the actor-critic. Timed in `src/trainer.py` around `train_component("actor_critic", ...)` (GPU-dominated, synchronized).
- `world_model_update`: time spent training the denoiser and reward/end model. Timed in `src/trainer.py` around `train_component("denoiser", ...)` and `train_component("rew_end_model", ...)` (GPU-dominated, synchronized).
- `distillation_oracle_query`: placeholder for future distillation/oracle calls (currently unused, so zero).

How the totals add up:
- The intent is for `epoch_wall_sec` to be approximately the sum of the above breakdown buckets, plus small residual overheads (logging, data loader iteration, Python bookkeeping, etc.).
- Diffusion sampling is tracked separately from imagination rollout to avoid double counting; `imagination_rollout` explicitly excludes diffusion sampling time.
- Counts for each key are logged as `timing/<key>_count` to indicate how many timing blocks contributed within the epoch.
