from typing import Dict, List, NamedTuple, Union

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


class TransitionRL(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: Dict
    info: Dict


class TransitionBC(NamedTuple):
    done: jnp.ndarray
    expert_action: jnp.ndarray
    obs: Dict
    info: Dict


def concatenate_transitions(
    transitions: List[Union[TransitionRL, TransitionBC]],
) -> Union[TransitionRL, TransitionBC]:
    dict_res = {}
    if isinstance(transitions[0], TransitionRL):
        list_attr = ["done", "action", "value", "reward", "log_prob", "obs", "info"]
    elif isinstance(transitions[0], TransitionBC):
        list_attr = ["done", "expert_action", "obs", "info"]
    else:
        raise ValueError("Unknown transition type of environment.")
    for key in list_attr:
        dict_res[key] = None

    for key in dict_res.keys():
        if getattr(transitions[0], key) is None:
            dict_res[key] = None
        elif isinstance(getattr(transitions[0], key), jnp.ndarray):
            dict_res[key] = jnp.concatenate(
                [getattr(trans, key)[None] for trans in transitions], axis=0
            )
        elif isinstance(getattr(transitions[0], key), Dict):
            dict_res[key] = concatenate_dicts(
                [getattr(trans, key) for trans in transitions]
            )

    res = TransitionRL(**dict_res)
    return res
