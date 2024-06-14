from typing import Optional, Callable
import torch

import ssm_rl.common.dense_nets as dn

nn = torch.nn
F = torch.nn.functional
dists = torch.distributions


class AbstractRSSMTM(nn.Module):

    def __init__(self,
                 action_dim: int,
                 build_with_obs_valid: bool,
                 obs_sizes: list[int]):
        super(AbstractRSSMTM, self).__init__()
        self._built_with_obs_valid = build_with_obs_valid
        self._action_dim = action_dim
        self._default_value = 0.0
        self._obs_sizes = obs_sizes

    @property
    def built_with_obs_valid(self) -> bool:
        return self._built_with_obs_valid

    @property
    def obs_sizes(self) -> list[int]:
        return self._obs_sizes

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @staticmethod
    def _stack_dicts(dicts: list[dict[str, torch.Tensor]], dim: int = 1) -> dict[str, torch.Tensor]:
        return {k: torch.stack([d[k] for d in dicts], dim=dim) for k in dicts[0].keys()}

    def predict(self,
                post_state: dict[str, torch.Tensor], action: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def update(self,
               prior_state: dict[str, torch.Tensor],
               obs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def update_with_obs_valid(self,
                              prior_state: dict[str, torch.Tensor],
                              obs: list[torch.Tensor],
                              obs_valid: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_initial(self, batch_size: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @property
    def feature_size(self):
        raise NotImplementedError

    def get_features(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def get_deterministic_features(self, state: dict[str, torch.Tensor]):
        raise NotImplementedError

    @property
    def hist_size(self):
        raise NotImplementedError

    @property
    def latent_size(self):
        raise NotImplementedError

    def get_next_posterior(self,
                           latent_obs: list[torch.Tensor],
                           action: torch.Tensor,
                           post_state: dict,
                           obs_valid: Optional[list[torch.Tensor]] = None) -> dict[str, torch.Tensor]:
        if obs_valid is None and self._built_with_obs_valid:
            obs_valid = [torch.ones(action.shape[0], 1, dtype=torch.bool, device=action.device) for _ in latent_obs]

        prior_state = self.predict(post_state=post_state, action=action)
        if obs_valid is None:
            post_state = self.update(prior_state=prior_state, obs=latent_obs)
        else:
            post_state = self.update_with_obs_valid(prior_state=prior_state, obs=latent_obs, obs_valid=obs_valid)
        return post_state

    def forward_pass(self,
                     embedded_obs: list[torch.Tensor],
                     actions: torch.Tensor,
                     obs_valid: Optional[list[torch.Tensor]] = None) \
            -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if obs_valid is None:
            return self._forward_pass(embedded_obs=embedded_obs, actions=actions)
        else:
            assert self._built_with_obs_valid, "Model was not built with obs_valid - Naive RSSM cannot switch"
            return self._forward_pass_with_obs_valid(embedded_obs=embedded_obs, actions=actions, obs_valid=obs_valid)

    def _forward_pass(self,
                      embedded_obs: list[torch.Tensor],
                      actions: torch.Tensor)\
            -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        batch_size, seq_length = embedded_obs[0].shape[:2]
        embedded_obs = [torch.unbind(lo, 1) for lo in embedded_obs]
        actions = torch.unbind(actions, 1)

        post_state = self.get_initial(batch_size=batch_size)

        prior_states = []
        post_states = []
        for i in range(seq_length):
            prior_state = self.predict(post_state=post_state,
                                       action=actions[i])
            post_state = self.update(prior_state=prior_state,
                                     obs=[lo[i] for lo in embedded_obs])

            prior_states.append(prior_state)
            post_states.append(post_state)
        return self._stack_dicts(post_states), self._stack_dicts(prior_states)

    def _forward_pass_with_obs_valid(self,
                                     embedded_obs: list[torch.Tensor],
                                     actions: torch.Tensor,
                                     obs_valid: list[torch.Tensor]):
        batch_size, seq_length = embedded_obs[0].shape[:2]
        embedded_obs = [torch.unbind(lo, 1) for lo in embedded_obs]
        obs_valid = [torch.unbind(ov, 1) for ov in obs_valid]

        actions = torch.unbind(actions, 1)

        post_state = self.get_initial(batch_size=batch_size)

        prior_states = []
        post_states = []

        for i in range(seq_length):
            prior_state = self.predict(post_state=post_state,
                                       action=actions[i])
            post_state = self.update_with_obs_valid(prior_state=prior_state,
                                                    obs=[lo[i] for lo in embedded_obs],
                                                    obs_valid=[ov[i] for ov in obs_valid])

            prior_states.append(prior_state)
            post_states.append(post_state)

        return self._stack_dicts(post_states), self._stack_dicts(prior_states)

    def open_loop_prediction(self,  initial_state: dict[str, torch.Tensor], actions: torch.Tensor)\
            -> dict[str, torch.Tensor]:
        state = initial_state
        action_list = torch.unbind(actions, 1)
        states = []
        for i, action in enumerate(action_list):
            state = self.predict(post_state=state, action=action)
            states.append(state)

        return self._stack_dicts(states)

    def rollout_policy(self,
                       state: dict,
                       policy_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                       num_steps: int) -> tuple[dict, torch.Tensor]:
        states = []
        log_probs = []
        for _ in range(num_steps):
            action, log_prob = policy_fn(self.get_features(state))
            state = self.predict(post_state=state, action=action)

            states.append(state)
            log_probs.append(log_prob)
        states = {k: torch.stack([s[k] for s in states], dim=1) for k in states[0].keys()}
        return states, torch.stack(log_probs, dim=1)

    @property
    def latent_distribution(self) -> str:
        raise NotImplementedError
