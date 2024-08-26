from typing import Dict, List, NamedTuple, Union
import jax.numpy as jnp
import numpy as np

from torchrl.data import PrioritizedReplayBuffer, ListStorage


class TransitionPPO(NamedTuple):
    done: Union[np.ndarray, jnp.ndarray]
    action: Union[np.ndarray, jnp.ndarray]
    value: Union[np.ndarray, jnp.ndarray]
    reward: Union[np.ndarray, jnp.ndarray]
    log_prob: Union[np.ndarray, jnp.ndarray]
    obs: Dict
    info: Dict


class TransitionBC(NamedTuple):
    done: Union[np.ndarray, jnp.ndarray]
    expert_action: Union[np.ndarray, jnp.ndarray]
    obs: Dict
    info: Dict


class TransitionDQN(NamedTuple):
    state: Dict
    action: Union[np.ndarray, jnp.ndarray]
    next_state: Dict
    reward: Union[np.ndarray, jnp.ndarray]
    done: Union[np.ndarray, jnp.ndarray]


class ReplayMemory:
    def __init__(self, size, alpha=0.0, beta=0):
        self.buffer = PrioritizedReplayBuffer(
            storage=ListStorage(size), alpha=alpha, beta=beta
        )

    def push(self, state, action, next_state, reward, done):
        batch_size = reward.shape[0]
        for batch in range(batch_size):
            single_state = {k: v[batch] for k, v in state.items()}
            single_next_state = {k: v[batch] for k, v in next_state.items()}
            transition = TransitionDQN(
                single_state,
                action[batch],
                single_next_state,
                reward[batch],
                done[batch],
            )
            self.buffer.add(transition)

    def sample(self, batch_size):
        samples, info = self.buffer.sample(batch_size=batch_size, return_info=True)
        return samples

    def update_priorities(self, indices, priorities):
        self.buffer.update_priorities(indices, priorities)

    def __len__(self):
        return len(self.buffer)


def concatenate_dicts(dict_list: List):
    dict_res = {}

    if not dict_list or all(not d for d in dict_list):
        return dict_res

    keys = set().union(*(d.keys() for d in dict_list))

    for key in keys:
        if isinstance(dict_list[0][key], np.ndarray):
            res = np.concatenate([d[key][None] for d in dict_list if d])
        else:
            res = np.array([d[key] for d in dict_list if d])
        dict_res[key] = res
    return dict_res


def concatenate_transitions(
    transitions: List[Union[TransitionPPO, TransitionBC, TransitionDQN]],
) -> Union[TransitionPPO, TransitionBC]:
    dict_res = {}
    if isinstance(transitions[0], TransitionPPO):
        list_attr = ["done", "action", "value", "reward", "log_prob", "obs", "info"]
    elif isinstance(transitions[0], TransitionBC):
        list_attr = ["done", "expert_action", "obs", "info"]
    elif isinstance(transitions[0], TransitionDQN):
        list_attr = ["done", "action", "reward", "state", "next_state"]
    else:
        raise ValueError("Unknown transition type of environment.")
    for key in list_attr:
        dict_res[key] = None

    for key in dict_res.keys():
        if getattr(transitions[0], key) is None:

            dict_res[key] = None
        elif isinstance(getattr(transitions[0], key), Union[np.ndarray, jnp.ndarray]):
            dict_res[key] = jnp.concatenate(
                [getattr(trans, key)[None] for trans in transitions], axis=0
            )
        elif isinstance(getattr(transitions[0], key), Dict):
            dict_res[key] = concatenate_dicts(
                [getattr(trans, key) for trans in transitions]
            )
    if isinstance(transitions[0], TransitionPPO):
        res = TransitionPPO(**dict_res)
    elif isinstance(transitions[0], TransitionBC):
        res = TransitionBC(**dict_res)
    elif isinstance(transitions[0], TransitionDQN):
        res = TransitionDQN(**dict_res)
    return res
