import torch
from ssm_rl.ssm_interface.abstract_ssm import AbstractSSM
from ssm_rl.rssm.transition.abstract_tm import AbstractRSSMTM
from typing import Optional, Callable, Union

nn = torch.nn


class RSSM(AbstractSSM):

    def __init__(self,
                 encoders: nn.ModuleList,
                 transition_model: AbstractRSSMTM,
                 reward_decoder: nn.Module):

        super(RSSM, self).__init__()

        self.encoders = encoders
        self.transition_model = transition_model
        self.reward_decoder = reward_decoder

    def encode(self,
               observations: list[torch.Tensor],
               obs_valid: Optional[list[torch.Tensor]] = None) -> list[torch.Tensor]:
        embedded_obs = [enc(observations[i], mask=None if obs_valid is None else obs_valid[i])
                        for i, enc in enumerate(self.encoders)]
        return embedded_obs

    @property
    def feature_size(self) -> int:
        return self.transition_model.feature_size

    @property
    def action_dim(self) -> int:
        return self.transition_model.action_dim

    def get_features(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.transition_model.get_features(state=state)

    def get_deterministic_features(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.transition_model.get_deterministic_features(state=state)

    def get_initial_state(self, batch_size: int) -> dict[str, torch.Tensor]:
        return self.transition_model.get_initial(batch_size=batch_size)

    @property
    def latent_distribution(self):
        return self.transition_model.latent_distribution

    def get_next_posterior(self,
                           observation: list[torch.Tensor],
                           action: torch.Tensor,
                           post_state: dict,
                           obs_valid: Optional[list[torch.Tensor]] = None) -> dict:
        observation = [torch.unsqueeze(obs, dim=1) for obs in observation]
        latent_obs = [lo.squeeze(dim=1) for lo in self.encode(observation, obs_valid=obs_valid)]
        return self.transition_model.get_next_posterior(latent_obs=latent_obs,
                                                        action=action,
                                                        post_state=post_state,
                                                        obs_valid=obs_valid)

    def predict_rewards_open_loop(self,
                                  initial_state: dict,
                                  actions: torch.Tensor):
        states = self.transition_model.open_loop_prediction(initial_state=initial_state,
                                                            actions=actions)
        return self.reward_decoder(self.transition_model.get_features(state=states))

    def rollout_policy(self,
                       state: dict,
                       policy_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                       num_steps: int) -> tuple[dict, torch.Tensor]:
        return self.transition_model.rollout_policy(state=state,
                                                    policy_fn=policy_fn,
                                                    num_steps=num_steps)

