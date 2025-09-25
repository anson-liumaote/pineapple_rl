# export LeggedRobotCfg and LeggedRobotCfgPPO to yaml (both a concise and a full recursive dump)
# 使用方式:
#   python legged_gym/legged_gym/scripts/export_config.py --target_dir <dir> [--task bipedal_him] [--no-minimal]
#
# 產物:
#   rl_cfg.yaml       (原本精簡版，可選擇關閉)
#   rl_full_cfg.yaml  (完整遞迴版)

import isaacgym  # noqa: F401
from isaacgym import gymapi
from legged_gym.envs import *          # noqa: F401,F403
from legged_gym.utils import task_registry
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.helpers import class_to_dict, parse_sim_params

import argparse
import os
import inspect
import yaml
import enum

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None


def cfg_obj_to_plain(obj, _visited=None):
    """
    遞迴轉換 config 物件為可 YAML 序列化的純 Python 結構。
    """
    if _visited is None:
        _visited = set()

    # --- 追加: numpy scalar / torch 特殊型別 / Enum 轉換 ---
    if np is not None and isinstance(obj, np.generic):
        return obj.item()
    if torch is not None:
        if isinstance(obj, (torch.device, torch.dtype)):
            return str(obj)
        # 單一 tensor 已在後面處理，這裡避免再走到 generic 區塊
    if isinstance(obj, enum.Enum):
        # 儘量回傳其值，若值本身不可序列化再回字串
        try:
            return obj.value
        except Exception:
            return str(obj)

    # 基本類型
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj

    # 已訪問避免循環
    obj_id = id(obj)
    if obj_id in _visited:
        return f"<circular_ref:{type(obj).__name__}>"
    _visited.add(obj_id)

    # 容器
    if isinstance(obj, (list, tuple, set)):
        return [cfg_obj_to_plain(v, _visited) for v in obj]
    if isinstance(obj, dict):
        return {k: cfg_obj_to_plain(v, _visited) for k, v in obj.items()}

    # NumPy / Torch
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    # 模組 / 類別 / 函數
    if inspect.ismodule(obj):
        return f"<module:{getattr(obj, '__name__', str(obj))}>"
    if inspect.isclass(obj):
        return f"<class:{obj.__name__}>"
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        return f"<callable:{getattr(obj, '__name__', 'anon')}>"

    # 其他自訂類別 (假設為 config)
    plain = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        if name == "__class__":
            continue
        try:
            val = getattr(obj, name)
        except Exception:
            continue
        if callable(val):
            continue
        # 過濾掉 gym handler 等較大的運行期物件
        if type(val).__name__ in ("Gym", "Viewer", "Sim", "Env", "TensorDeviceAllocator"):
            plain[name] = f"<omitted:{type(val).__name__}>"
            continue
        plain[name] = cfg_obj_to_plain(val, _visited)
    # 若空 dict 且不是明顯的 config，給個字串以利除錯
    if not plain:
        return repr(obj)
    return plain


def export_minimal(env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO, env: LeggedRobot, jit_script_path: str, target_dir: str):
    yaml_file_name = 'rl_cfg.yaml'
    yaml_file_path = os.path.join(target_dir, yaml_file_name)
    save_dict = {
        'env': {
            'num_actions': env_cfg.env.num_actions,
            'num_observations': env_cfg.env.num_observations,
            'gym_joint_names': env.dof_names,
        },
        'commands': {
            'ranges': {
                'lin_vel_x': env_cfg.commands.ranges.lin_vel_x,
                'lin_vel_y': env_cfg.commands.ranges.lin_vel_y,
                'ang_vel_yaw': getattr(env_cfg.commands.ranges, 'ang_vel_yaw', None),
            }
        },
        'init_state': {
            'default_joint_angles': env_cfg.init_state.default_joint_angles,
        },
        'control': {
            'control_type': env_cfg.control.control_type,
            'action_scale': getattr(env_cfg.control, 'action_scale', None),
            'stiffness': getattr(env_cfg.control, 'stiffness', None),
            'damping': getattr(env_cfg.control, 'damping', None),
        },
        'normalization': {
            'obs_scales': {
                'lin_vel': env_cfg.normalization.obs_scales.lin_vel,
                'ang_vel': env_cfg.normalization.obs_scales.ang_vel,
                'dof_pos': env_cfg.normalization.obs_scales.dof_pos,
                'dof_vel': env_cfg.normalization.obs_scales.dof_vel,
                'height_measurements': env_cfg.normalization.obs_scales.height_measurements,
            },
            'clip_observations': env_cfg.normalization.clip_observations,
            'clip_actions': env_cfg.normalization.clip_actions,
        },
        'rewards': {
            'scales': cfg_obj_to_plain(env_cfg.rewards.scales),
            'clip_observations': env_cfg.normalization.clip_observations,
            'clip_actions': env_cfg.normalization.clip_actions,
        },
        'jit_script_path': jit_script_path,
    }
    wrapped = {'rl_config': save_dict}
    with open(yaml_file_path, 'w') as f:
        f.write("# This file is generated by export_config.py (minimal subset)\n")
        f.write("# It is not recommended to modify this file manually\n")
        yaml.safe_dump(wrapped, f, sort_keys=False)
    print(f"[INFO] Wrote minimal config to {yaml_file_path}")


def export_full(env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO, env: LeggedRobot, jit_script_path: str, target_dir: str):
    yaml_file_name = 'rl_full_cfg.yaml'
    yaml_file_path = os.path.join(target_dir, yaml_file_name)

    full_env_cfg = cfg_obj_to_plain(env_cfg)
    full_train_cfg = cfg_obj_to_plain(train_cfg)

    payload = {
        'env_cfg': full_env_cfg,
        'train_cfg': full_train_cfg,
        'derived': {
            'dof_names': env.dof_names,
            'num_dofs': len(env.dof_names),
            'jit_script_path': jit_script_path,
        }
    }

    with open(yaml_file_path, 'w') as f:
        f.write("# Auto-generated full configuration export\n")
        yaml.safe_dump(payload, f, sort_keys=False)
    print(f"[INFO] Wrote full config to {yaml_file_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='bipedal_him')
    parser.add_argument('--target_dir', type=str, required=True)
    parser.add_argument('--no-minimal', action='store_true', help="不輸出精簡版 rl_cfg.yaml 只輸出完整版")
    args = parser.parse_args()

    task_name = args.task
    target_dir = args.target_dir

    if not os.path.exists(target_dir):
        raise ValueError(f"target_dir {target_dir} does not exist")

    if task_name not in task_registry.task_classes:
        raise ValueError(f"Task with name: {task_name} was not registered")

    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg: LeggedRobotCfg
    train_cfg: LeggedRobotCfgPPO

    task_class = task_registry.get_task_class(name=task_name)
    # 解析 sim 參數
    sim_params = {"sim": class_to_dict(env_cfg.sim)}
    # 補齊 parse_sim_params 所需欄位
    args.physics_engine = gymapi.SIM_PHYSX
    args.use_gpu = True
    args.subscenes = 0
    args.use_gpu_pipeline = True
    args.num_threads = 0
    sim_params = parse_sim_params(args, sim_params)

    # 建立環境
    env = task_class(env_cfg, sim_params, gymapi.SIM_PHYSX, 'cuda', True)
    env: LeggedRobot

    # 推論策略路徑
    jit_script_path = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        'logs',
        train_cfg.runner.experiment_name,
        'exported',
        'policies',
        'policy_1.pt'
    )

    # 匯出
    if not args.no_minimal:
        export_minimal(env_cfg, train_cfg, env, jit_script_path, target_dir)
    export_full(env_cfg, train_cfg, env, jit_script_path, target_dir)


if __name__ == "__main__":
    main()


