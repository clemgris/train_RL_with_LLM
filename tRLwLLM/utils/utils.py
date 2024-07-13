from typing import Dict, List, NamedTuple

import jax.numpy as jnp
import numpy as np


def concatenate_dicts(dict_list: List):
    dict_res = {}

    if not dict_list or all(not d for d in dict_list):
        return dict_res

    keys = set().union(*(d.keys() for d in dict_list))

    for key in keys:
        if isinstance(dict_list[0][key], np.ndarray):
            res = jnp.concatenate([d[key][None] for d in dict_list if d])
        else:
            res = np.array([d[key] for d in dict_list if d])
        dict_res[key] = res
    return dict_res


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: Dict
    info: Dict


def concatenate_transitions(transitions: List[Transition]) -> Transition:
    dict_res = {
        "done": None,
        "action": None,
        "value": None,
        "reward": None,
        "log_prob": None,
        "obs": None,
        "info": None,
    }

    dict_res["done"] = jnp.concatenate(
        [trans.done[None] for trans in transitions], axis=0
    )
    dict_res["action"] = jnp.concatenate(
        [trans.action[None] for trans in transitions], axis=0
    )
    dict_res["value"] = jnp.concatenate(
        [trans.value[None] for trans in transitions], axis=0
    )
    dict_res["reward"] = jnp.concatenate(
        [trans.reward[None] for trans in transitions], axis=0
    )
    dict_res["log_prob"] = jnp.concatenate(
        [trans.log_prob[None] for trans in transitions], axis=0
    )
    dict_res["obs"] = concatenate_dicts([trans.obs for trans in transitions])
    dict_res["info"] = concatenate_dicts([trans.info for trans in transitions])

    return Transition(**dict_res)
