"""
Export CMG Student V2 policy to TorchScript JIT.

The exported model accepts raw (un-normalized) obs of shape (1, 1237) and outputs
actions of shape (1, 23).  The normalizer is baked in so the deploy script
does NOT need a separate normalizer step.

Usage:
    cd legged_gym/legged_gym/scripts
    python save_jit_cmg_stu_v2.py \
        --proj_name g1_cmg_stu_v2 \
        --exptid    cmg_stu_v4   \
        --checkpoint -1           \
        --device cpu
"""

import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../rsl_rl"))

import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic_mimic import Actor, get_activation
from rsl_rl.utils.normalizer import Normalizer


# ── network constants (must match G1MimicCMGStuV2Cfg) ─────────────────────────
NUM_OBS            = 1237   # 1160 (priv_mimic) + 77 (proprio)
NUM_MIMIC_OBS      = 1160   # 20 steps × 58 dims
NUM_MOTION_STEPS   = 20
MOTION_LATENT_DIM  = 128
NUM_ACTIONS        = 23
ACTOR_HIDDEN_DIMS  = [512, 512, 256, 128]
ACTIVATION         = "silu"
LAYER_NORM         = True


# ── wrapper: normalizer + actor ───────────────────────────────────────────────
class CMGStuV2Policy(nn.Module):
    """TorchScript-exportable wrapper: normalize obs then run actor."""

    def __init__(self, actor: Actor, normalizer):
        super().__init__()
        self.actor = actor
        # bake normalizer state as plain buffers (JIT-friendly)
        # Normalizer uses _mean, _std, _eps, _clip
        self.register_buffer("obs_mean", normalizer._mean.clone())
        self.register_buffer("obs_std",  normalizer._std.clone())
        self.obs_eps  = float(normalizer._eps)
        self.obs_clip = float(normalizer._clip) if normalizer._clip != float("inf") else 1e9

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_norm = (obs - self.obs_mean) / (self.obs_std + self.obs_eps)
        obs_norm = torch.clamp(obs_norm, -self.obs_clip, self.obs_clip)
        return self.actor(obs_norm)


# ── helpers ───────────────────────────────────────────────────────────────────
def get_load_path(root, checkpoint=-1):
    if checkpoint == -1:
        models = [f for f in os.listdir(root) if f.startswith("model_") and f.endswith(".pt")]
        models.sort(key=lambda m: int(m.split("_")[-1].split(".")[0]))
        model = models[-1]
    else:
        model = f"model_{checkpoint}.pt"
    return os.path.join(root, model), int(model.split("_")[-1].split(".")[0])


def build_actor(device):
    activation = get_activation(ACTIVATION)
    actor = Actor(
        num_observations       = NUM_OBS,
        num_motion_observations= NUM_MIMIC_OBS,
        num_motion_steps       = NUM_MOTION_STEPS,
        motion_latent_dim      = MOTION_LATENT_DIM,
        num_actions            = NUM_ACTIONS,
        actor_hidden_dims      = ACTOR_HIDDEN_DIMS,
        activation             = activation,
        layer_norm             = LAYER_NORM,
        tanh_encoder_output    = False,
    ).to(device)
    return actor


def main(args):
    device = torch.device(args.device)
    load_root = os.path.join(
        os.path.dirname(__file__), "../../logs", args.proj_name, args.exptid
    )
    load_path, ckpt_num = get_load_path(load_root, args.checkpoint)
    print(f"Loading checkpoint: {load_path}")

    ckpt = torch.load(load_path, map_location=device, weights_only=False)

    # ── build actor & load weights ────────────────────────────────────────────
    actor = build_actor(device)
    # model_state_dict contains the full ActorCriticMimic; extract actor sub-keys
    full_sd = ckpt["model_state_dict"]
    actor_sd = {k[len("actor."):]: v for k, v in full_sd.items() if k.startswith("actor.")}
    actor.load_state_dict(actor_sd)
    actor.eval()

    # ── load normalizer ───────────────────────────────────────────────────────
    norm_obj = ckpt.get("normalizer")
    if norm_obj is None:
        raise RuntimeError("Checkpoint has no 'normalizer' key – cannot bake normalizer.")
    if isinstance(norm_obj, Normalizer):
        normalizer = norm_obj.to(device)
    else:
        # saved as state_dict
        normalizer = Normalizer(NUM_OBS, device=device)
        normalizer.load_state_dict(norm_obj)
        normalizer = normalizer.to(device)
    normalizer.eval()

    # ── wrap and export ───────────────────────────────────────────────────────
    policy = CMGStuV2Policy(actor, normalizer).to(device)
    policy.eval()

    traced_dir = os.path.join(load_root, "traced")
    os.makedirs(traced_dir, exist_ok=True)

    dummy_obs = torch.zeros(1, NUM_OBS, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(policy, dummy_obs)

    save_path = os.path.join(traced_dir, f"{args.exptid}-{ckpt_num}-jit.pt")
    traced.save(save_path)
    print(f"Saved JIT policy → {os.path.abspath(save_path)}")

    # quick sanity check
    with torch.no_grad():
        out = traced(dummy_obs)
    print(f"Output shape: {out.shape}  (expected [1, {NUM_ACTIONS}])")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj_name",  type=str, default="g1_cmg_stu_v2")
    parser.add_argument("--exptid",     type=str, required=True)
    parser.add_argument("--checkpoint", type=int, default=-1,
                        help="-1 = latest checkpoint")
    parser.add_argument("--device",     type=str, default="cpu",
                        help="cpu or cuda:0")
    args = parser.parse_args()
    main(args)
