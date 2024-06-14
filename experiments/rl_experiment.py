import torch
from collections import OrderedDict
from typing import Optional
import numpy as np
import random
import warnings

from ssm_rl.util.config_dict import ConfigDict
import ssm_rl.model_learning.evaluation.model_eval_factories as mef


class RLExperiment:

    def __init__(self,
                 env_factory,
                 model_factory,
                 policy_factory,
                 trainer_factory,
                 rl_factory,
                 seed: int,
                 verbose: bool,
                 save_path: Optional[str] = None,
                 model_eval_factories=None,
                 policy_eval_factories=None,
                 use_cuda_if_available: bool = True,
                 fully_deterministic: bool = False):

        self._verbose = verbose
        self._seed = seed
        self._save_path = save_path
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if fully_deterministic:
            warnings.warn("Fully Deterministic run requested... this will be slower!")
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        self._conf_save_dict = {}

        self._env_factory = env_factory
        self._model_factory = model_factory
        self._policy_factory = policy_factory
        self._trainer_factory = trainer_factory
        self._rl_factory = rl_factory
        self._model_eval_factories = [] if model_eval_factories is None else model_eval_factories
        self._policy_eval_factories = [] if policy_eval_factories is None else policy_eval_factories

        self._built = False
        self._env = None
        self._model = None
        self._policy = None
        self._trainer = None
        self._model_evaluators = None
        self._policy_evaluators = None
        self._replay_buffer = None
        self._data_collector = None
        self._rb_kwargs_train = None
        self._rb_kwargs_val = None

    def get_default_params(self):
        defaults = OrderedDict(env=self._env_factory.get_default_config(),
                               model=self._model_factory.get_default_config(),
                               policy=self._policy_factory.get_default_config(),
                               trainer=self._trainer_factory.get_default_config(),
                               rl=self._rl_factory.get_default_config())
        if self._model_eval_factories is not None:
            model_eval_conf = ConfigDict()
            for factory in self._model_eval_factories:
                model_eval_conf.add_subconf(factory.name(), factory.get_default_config())
            model_eval_conf.finalize_adding()
            defaults["model_eval"] = model_eval_conf
        if self._policy_eval_factories is not None:
            policy_eval_conf = ConfigDict()
            for factory in self._policy_eval_factories:
                policy_eval_conf.add_subconf(factory.name(), factory.get_default_config())
            policy_eval_conf.finalize_adding()
            defaults["policy_eval"] = policy_eval_conf
        return defaults

    def build(self,
              env_config: Optional[ConfigDict] = None,
              model_config: Optional[ConfigDict] = None,
              policy_config: Optional[ConfigDict] = None,
              rl_config: Optional[ConfigDict] = None,
              trainer_config: Optional[ConfigDict] = None,
              model_eval_config: Optional[dict[Optional[ConfigDict]]] = None,
              policy_eval_config: Optional[dict[Optional[ConfigDict]]] = None):

        # Env
        if env_config is None:
            env_config = self._env_factory.get_default_config()
        self._conf_save_dict["env"] = env_config
        if self._verbose:
            print("=== Environment ===")
            print(env_config)

        image_to_info = any(m.needs_image_in_info for m in self._model_eval_factories + self._policy_eval_factories)
        state_to_info = any(m.needs_state_in_info for m in self._model_eval_factories + self._policy_eval_factories)

        base_env = self._env_factory.build(seed=self._seed,
                                           config=env_config,
                                           image_to_info=image_to_info,
                                           full_state_to_info=state_to_info)
        obs_types = ["image" if obs_is_image else "vector" for obs_is_image in base_env.obs_are_images]
        # MBRL Stuff
        if rl_config is None:
            rl_config = self._rl_factory.get_default_config()
        self._conf_save_dict["mbrl"] = rl_config
        if self._verbose:
            print("=== MBRL ===")
            print(rl_config)

        if any(base_env.obs_are_images):
            img_sizes = [o.shape for o, oai in zip(base_env.observation_space, base_env.obs_are_images) if oai]
            img_preprocessor = self._rl_factory.build_img_preprocessor(img_sizes=img_sizes,
                                                                       config=rl_config.img_preprocessing)
            obs_sizes = [o.shape[0] if len(o.shape) == 1 else o.shape for o in base_env.observation_space]
            for i, os in enumerate(obs_sizes):
                if base_env.obs_are_images[i]:
                    obs_sizes[i] = (os[0], ) + img_preprocessor.output_img_size
        else:
            img_preprocessor = None
            obs_sizes = [o.shape[0] if len(o.shape) == 1 else o.shape for o in base_env.observation_space]
        self._env, self._replay_buffer = \
            self._rl_factory.build_env_and_replay_buffer(env=base_env,
                                                         img_preprocessor=img_preprocessor,
                                                         config=rl_config)
        self._rb_kwargs_train = {"seq_length": rl_config.rl_exp.model_updt_seq_length,
                                 "batch_size": rl_config.rl_exp.model_updt_batch_size,
                                 "num_batches": rl_config.rl_exp.model_updt_steps,
                                 "val_dataset": False}
        if rl_config.rl_exp.bypass_mask in [None, "none"]:
            bypass_mask = [False] * len(obs_sizes)
        else:
            bypass_mask = rl_config.rl_exp.bypass_mask
            for i, bypass in enumerate(bypass_mask):
                if bypass:
                    assert not base_env.obs_are_images[i], "Bypassing images is not supported"
                    if i < len(self._model_factory.encoder_factories):
                        self._model_factory.encoder_factories[i] = None
                    if i < len(self._trainer_factory.objective_factory.observation_loss_factories):
                        self._trainer_factory.objective_factory.observation_loss_factories[i] = None
                    encoder_key = f"encoder{i}"
                    obs_key = f"obs{i}"
                    if encoder_key in model_config.subconfig_names():
                        model_config.remove_subconf(encoder_key)
                    if obs_key in trainer_config.objective.subconfig_names():
                        trainer_config.objective.remove_subconf(obs_key)
            self._model_factory.encoder_factories =\
                [ef for ef in self._model_factory.encoder_factories if ef is not None]
            self._trainer_factory.objective_factory.observation_loss_factories = \
                [olf for olf in self._trainer_factory.objective_factory.observation_loss_factories if olf is not None]

        assert not any([(is_image and bypass) for is_image, bypass in zip(base_env.obs_are_images, bypass_mask)]), \
            "Cannot bypass image observations"

        # Model
        if model_config is None:
            model_config = self._model_factory.get_default_config()
        self._conf_save_dict["model"] = model_config
        if self._verbose:
            print("=== MODEL ===")
            print(model_config)

        with_obs_valid = self._replay_buffer.has_obs_valid
        self._model = self._model_factory.build(config=model_config,
                                                input_sizes=obs_sizes,
                                                input_types=obs_types,
                                                bypass_mask=bypass_mask,
                                                action_dim=base_env.action_space.shape[0],
                                                with_obs_valid=with_obs_valid).to(self._device)
        # Policy
        if policy_config is None:
            policy_config = self._policy_factory.get_default_config()
        self._conf_save_dict["policy"] = policy_config
        if self._verbose:
            print("=== Policy ===")
            print(policy_config)

        self._policy = self._policy_factory.build(model=self._model,
                                                  obs_are_images=base_env.obs_are_images,
                                                  obs_sizes=obs_sizes,
                                                  bypass_mask=bypass_mask,
                                                  img_preprocessor=img_preprocessor,
                                                  config=policy_config,
                                                  action_space=base_env.action_space,
                                                  device=self._device)
        self._data_collector = self._rl_factory.build_data_collector(env=self._env,
                                                                     policy=self._policy,
                                                                     config=rl_config)

        # Trainer
        if trainer_config is None:
            trainer_config = self._trainer_factory.get_default_config()
        self._conf_save_dict["trainer"] = trainer_config
        if self._verbose:
            print("=== Policy Trainer ===")
            print(trainer_config)
        self._trainer = \
            self._trainer_factory.build(policy=self._policy,
                                        model=self._model,
                                        target_sizes=obs_sizes,
                                        target_types=obs_types,
                                        config=trainer_config,
                                        device=self._device)

        self._model_evaluators = []
        if self._model_eval_factories is not None and len(self._model_eval_factories) > 0:  # and all(mf is not None for mf in self._model_eval_factories):
            if model_eval_config is None:
                model_eval_config = ConfigDict()
            for factory in self._model_eval_factories:
                if factory.name() not in model_eval_config.subconfig_names():
                    model_eval_config.add_subconf(factory.name(), factory.get_default_config())
                config = getattr(model_eval_config, factory.name())
                evaluator = factory.build(model_objective=self._trainer.model_objective,
                                          save_path=self._save_path,
                                          config=config,
                                          fixed_eval_buffer=None,
                                          device=self._device).to(self._device)
                if self._verbose:
                    print("=== {} ===".format(evaluator.name()))
                    print(config)
                self._model_evaluators.append(evaluator)
            self._conf_save_dict["model_eval"] = model_eval_config

        self._policy_evaluators = []
        if self._policy_eval_factories is not None:
            if policy_eval_config is None:
                policy_eval_config = ConfigDict()
            for factory in self._policy_eval_factories:
                if factory.name() not in policy_eval_config.subconfig_names():
                    policy_eval_config.add_subconf(factory.name(), factory.get_default_config())
                config = getattr(policy_eval_config, factory.name())
                evaluator = factory.build(env=self._env,
                                          policy=self._policy,
                                          save_path=self._save_path,
                                          config=config).to(self._device)
                if self._verbose:
                    print("=== {} ===".format(evaluator.name()))
                    print(config)
                self._policy_evaluators.append(evaluator)
            self._conf_save_dict["policy_eval"] = policy_eval_config
        self._built = True
        return self._conf_save_dict

    def iterate(self, iteration: int):

        assert self._built
        if self._verbose:
            print("=== Iteration {:04d} ===".format(iteration))
        train_loader = self._replay_buffer.get_data_loader(device=self._device, **self._rb_kwargs_train)
        train_dict, train_time = self._trainer.train_epoch(data_loader=train_loader,
                                                           env_step=self._data_collector.global_step,
                                                           mode=None)
        with torch.no_grad():
            log_dict = self._model_log(model_log_dict=train_dict,
                                       time=train_time)
        if self._model is not None:
            log_dict, val_loader = self._eval_model(iteration=iteration,
                                                    log_dict=log_dict)
        with torch.no_grad():
            log_dict = self._collect_new_data(iteration=iteration,
                                              log_dict=log_dict)
            log_dict = self._eval_policy(iteration=iteration,
                                         log_dict=log_dict)
            return log_dict

    def _eval_model(self,
                    iteration: int,
                    log_dict: OrderedDict):
        dataset_needed = \
            self._model_evaluators is not None and any(me.will_evaluate(iteration) for me in self._model_evaluators)

        if dataset_needed:
            val_loader = self._replay_buffer.get_data_loader(device=self._device,
                                                             **self._rb_kwargs_val)
        else:
            val_loader = None
        if self._model_evaluators is not None:
            eval_results = []
            for evaluator in self._model_evaluators:
                eval_results.append(evaluator.evaluate(data_loader=val_loader,
                                                       iteration=iteration))
            log_dict = self._log_evaluators(eval_results=eval_results,
                                            evaluators=self._model_evaluators,
                                            log_dict=log_dict)
            return log_dict, val_loader
        else:
            return log_dict, None

    def _eval_policy(self,
                     iteration: int,
                     log_dict: OrderedDict):
        if self._policy_evaluators is not None:
            eval_results = []
            for evaluator in self._policy_evaluators:
                eval_results.append(evaluator.evaluate(iteration=iteration))
            log_dict = self._log_evaluators(eval_results=eval_results,
                                            evaluators=self._policy_evaluators,
                                            log_dict=log_dict)
        return log_dict
    
    def _collect_new_data(self,
                          iteration: int,
                          log_dict: OrderedDict):

        assert not self._replay_buffer.is_frozen
        observations, actions, rewards, terminated, truncated, infos, collect_time = self._data_collector.collect()
        avg_reward = sum([torch.sum(r).detach().numpy() for r in rewards]) / len(rewards)
        avg_seq_length = sum([len(r) for r in rewards]) / len(rewards)
        if "success" in infos[0].keys():
            avg_success = sum([torch.max(i["success"]).detach().numpy() for i in infos]) / len(infos)
        else:
            avg_success = None
        log_dict = self._log_collection(num_seqs=len(rewards),
                                        avg_reward=avg_reward,
                                        avg_success=avg_success,
                                        avg_seq_length=avg_seq_length,
                                        collect_time=collect_time,
                                        log_dict=log_dict)

        self._replay_buffer.add_data(obs=observations,
                                     actions=actions,
                                     rewards=rewards,
                                     terminated=terminated,
                                     truncated=truncated,
                                     infos=infos)
        return log_dict

    def save_replay_buffer(self, path: str):
        self._replay_buffer.save_to_disk(path=path)

    def _model_log(self,
                   model_log_dict: dict,
                   time: Optional[float] = None,
                   log_dict: Optional[OrderedDict] = None,
                   training=True) -> OrderedDict:
        prefix, long_str = ("train", "Training") if training else ("eval", "Validation")
        if self._verbose:
            log_str = "Model {}, ".format(long_str)
            for k, v in model_log_dict.items():
                if np.isscalar(v):
                    log_str += "{}: {:.5f} ".format(k, v)
            if time is not None:
                log_str += "Took {:.3f} seconds".format(time)
            print(log_str)
        if log_dict is None:
            log_dict = OrderedDict()
        for k, v in model_log_dict.items():
            if "/" in k:
                log_dict[k.replace("/", "_{}/".format(prefix))] = v
            else:
                log_dict["{}/{}".format(prefix, k)] = v
        if time is not None:
            log_dict["{}/time".format(prefix)] = time
        return log_dict

    def _log_evaluators(self,
                        eval_results: list,
                        evaluators: list,
                        log_dict: Optional[OrderedDict] = None) -> OrderedDict:
        if eval_results is not None:
            if self._verbose:
                for results, evaluator in zip(eval_results, evaluators):
                    log_str = "{}: ".format(evaluator.name())
                    if results is not None:
                        for k, v in results.items():
                            log_str += self.log_str_form_kv_pair(k, v)
                        print(log_str)
            if log_dict is None:
                log_dict = OrderedDict()
            for results, evaluator in zip(eval_results, evaluators):
                if results is not None:
                    for k, v in results.items():
                        log_dict["{}/{}".format(evaluator.name(), k)] = v
            return log_dict

    @staticmethod
    def log_str_form_kv_pair(k, v) -> str:
        if isinstance(v, float):
            log_str = "{}: {:.5f} ".format(k, v)
        elif isinstance(v, np.ndarray):
            log_str = "{}: {} ".format(k,
                                       np.array2string(v, precision=5, max_line_width=int(1e300)))
        else:
            log_str = "{}: {} ".format(k, str(v))
        return log_str

    def _log_collection(self,
                        num_seqs: int,
                        avg_reward: float,
                        avg_success: Optional[float],
                        avg_seq_length: float,
                        collect_time: float,
                        log_dict: Optional[OrderedDict] = None) -> OrderedDict:
        if self._verbose:
            collect_log_str = "Data Collection: Collected {:03d} Sequence(s) ".format(num_seqs)
            collect_log_str += "with average reward of {:.5f} ".format(avg_reward)
            collect_log_str += "and average length of {:.2f} ".format(avg_seq_length)
            if avg_success is not None:
                collect_log_str += "and average success of {:.5f} ".format(avg_success)
            collect_log_str += "Took {:.3f} seconds.".format(collect_time)
            print(collect_log_str)
        if log_dict is None:
            log_dict = OrderedDict()
        log_dict["collect/num_seqs"] = num_seqs
        log_dict["collect/avg_len"] = avg_seq_length
        log_dict["collect/avg_reward"] = avg_reward
        if avg_success is not None:
            log_dict["collect/avg_success"] = avg_success
        log_dict["collect/time"] = collect_time

        return log_dict
