import torch
from pose.utils.cmg_motion_lib import CMGMotionLib
from legged_gym import LEGGED_GYM_ROOT_DIR


def run_smoke():
    device = 'cpu'
    num_envs = 2
    cmg_model_path = f"{LEGGED_GYM_ROOT_DIR}/../cmg_workspace/runs/cmg_20260123_194851/cmg_final.pt"
    cmg_data_path = f"{LEGGED_GYM_ROOT_DIR}/../cmg_workspace/dataloader/cmg_training_data.pt"
    urdf_path = f"{LEGGED_GYM_ROOT_DIR}/../assets/g1/g1_custom_collision_with_fixed_hand.urdf"

    lib = CMGMotionLib(cmg_model_path=cmg_model_path,
                       cmg_data_path=cmg_data_path,
                       urdf_path=urdf_path,
                       device=device,
                       num_envs=num_envs,
                       episode_length_s=2.0,
                       dt=0.02)

    env_ids = torch.tensor([0, 1], dtype=torch.long)
    lib.reset(env_ids)

    # basic properties
    print('trajectory_buffer shape:', lib._trajectory_buffer.shape)
    print('root_pos_buffer shape:', lib._root_pos_buffer.shape)
    print('root_rot_buffer shape:', lib._root_rot_buffer.shape)

    # check current frame retrieval
    motion_ids = torch.tensor([0, 1], device=device)
    motion_times = torch.zeros((2,), device=device)
    root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos = lib.calc_motion_frame(motion_ids, motion_times)
    print('current dof_pos shape', dof_pos.shape)
    print('current body_pos shape', body_pos.shape)

    # tiled future query: ask for 5 future steps (tiled order like _get_mimic_obs)
    num_steps = 5
    motion_times_tiled = (torch.arange(num_steps) * lib._dt).unsqueeze(0).repeat(num_envs, 1).flatten()
    motion_ids_tiled = motion_ids.unsqueeze(-1).repeat(1, num_steps).flatten()

    rp, rr, rv, rav, dp, dv, bp = lib.calc_motion_frame(motion_ids_tiled, motion_times_tiled)
    print('tiled root_pos shape', rp.shape)
    print('tiled dof_pos shape', dp.shape)
    print('tiled body_pos shape', bp.shape)

    # step a few times and verify _motion_times updated and buffer idx advanced
    before_times = lib._motion_times.clone()
    lib.step()
    lib._update_root_state(lib._dt)
    after_times = lib._motion_times.clone()
    print('motion_times before', before_times)
    print('motion_times after', after_times)

    # check mapping from 29->23 dims
    sample_norm = lib._trajectory_buffer[0, 0]
    denorm = lib._denormalize_motion(sample_norm.unsqueeze(0))
    pos23, vel23 = lib._map_29_to_23(denorm)
    assert pos23.shape[-1] == 23
    print('29->23 mapping ok, pos shape', pos23.shape)

    print('SMOKE TEST PASSED')


if __name__ == '__main__':
    run_smoke()
