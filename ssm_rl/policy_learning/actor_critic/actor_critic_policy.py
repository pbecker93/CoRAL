import torch
from ssm_rl.policy_learning.common.img_preprocessor import AbstractImgPreprocessor
from ssm_rl.policy_learning.common.bypassing import BypassPolicy

nn = torch.nn
opt = torch.optim


class ActorCriticPolicy(BypassPolicy):

    def __init__(self,
                 model,
                 actor,
                 critic,
                 action_dim: int,
                 obs_are_images: list[bool],
                 img_preprocessor: AbstractImgPreprocessor,
                 use_deterministic_features: bool,
                 bypass_mask: list[bool],
                 device: torch.device):
        super(ActorCriticPolicy, self).__init__(model=model,
                                                actor=actor,
                                                action_dim=action_dim,
                                                obs_are_images=obs_are_images,
                                                img_preprocessor=img_preprocessor,
                                                use_deterministic_features=use_deterministic_features,
                                                bypass_mask=bypass_mask,
                                                device=device)
        self.critic = critic
