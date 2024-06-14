import os
from cw2.experiment import AbstractIterativeExperiment
from cw2.cw_data import cw_logging
from cw2.cw_error import ExperimentSurrender

from experiments.cw_util.printer_logger import PrintLogger

from ssm_rl.util.config_dict import ConfigDict
from experiments.rl_experiment import RLExperiment


class Cw2Experiment(AbstractIterativeExperiment):

    IGNORE_KEYS = ["experiment", "encoder1", "obs1"]

    def __init__(self):
        super(Cw2Experiment, self).__init__()
        self._experiment = None
        self._printer_logger = None
        self._log_path = None
        self._save_replay_buffer = False

    @staticmethod
    def check_and_update(key: str,
                         params: dict,
                         conf_dict: ConfigDict,
                         ignore_keys: list[str] = []):

        val_key_list = list(conf_dict.keys()) + list(conf_dict.subconfig_names()) + \
                       ([] if ignore_keys is None else ignore_keys)
        for k in params.keys():
            if k not in val_key_list:
                raise AssertionError(f"Unused key: {k}")
        try:
            conf_dict.rec_update(params, ignore_keys=ignore_keys)
        except AssertionError as err:
            raise AssertionError(f"Update failed for {key}: {err}") from err

    @staticmethod
    def default_experiment_config():
        raise NotImplementedError

    def setup_experiment(self,
                         exp_config: ConfigDict,
                         seed_offset: int = 0,
                         save_path: str = None):
        raise NotImplementedError

    @property
    def save_interval(self) -> int:
        return -1

    def initialize(self, config: dict, rep: int, logger: cw_logging.AbstractLogger) -> None:
        try:
            from cw2.cw_data.cw_wandb_logger import WandBLogger
            from experiments.cw_util.wb_logger import CustomWandBLogger
            has_wandb = True
        except ImportError:
            has_wandb = False

        self._log_path = config["_rep_log_path"]

        params = config.get("params", None)

        exp_config = self.default_experiment_config()
        if "experiment" in params.keys():
            self._save_replay_buffer = params["experiment"].pop("save_replay_buffer", False)

        if params is not None and "experiment" in params.keys():
            self.check_and_update(key="experiment",
                                  params=params["experiment"],
                                  conf_dict=exp_config)
        self._experiment = self.setup_experiment(exp_config,
                                                 seed_offset=rep,
                                                 save_path=self._log_path)

        conf_dict = self._experiment.get_default_params()
        for k in params.keys():
            assert k in conf_dict.keys() or k in Cw2Experiment.IGNORE_KEYS, "Unused key: {}".format(k)

        for k, v in conf_dict.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if params is not None and params.get(k, None) is not None and kk in params[k].keys():
                        self.check_and_update(key=k,
                                              params=params[k][kk],
                                              conf_dict=vv,
                                              ignore_keys=Cw2Experiment.IGNORE_KEYS)
            elif params is not None and k in params.keys():
                self.check_and_update(key=k,
                                      params=params[k],
                                      conf_dict=v,
                                      ignore_keys=Cw2Experiment.IGNORE_KEYS)

        kwargs = {}
        for k, v in conf_dict.items():
            kwargs["{}_config".format(k)] = v
        build_conf = self._experiment.build(**kwargs)

        for l in logger:
            if isinstance(l, PrintLogger):
                self._printer_logger = l
            if has_wandb and (isinstance(l, WandBLogger) or isinstance(l, CustomWandBLogger)):
                rd = {"experiment": exp_config.get_raw_dict()} | {k: v.get_raw_dict() for k, v in build_conf.items()}
                if l.run is not None:
                    l.run.config.update(rd, allow_val_change=True)
        if self._printer_logger is not None:
            self._printer_logger.preprocess(log_dict={"Experiment": exp_config, **build_conf})

    def iterate(self, config: dict, rep: int, n: int) -> dict:
        # how convenient
        log_dict = self._experiment.iterate(n)
        if self.save_interval > 0 and n % self.save_interval == 0:
            path = os.path.join(self._log_path, "experiment_at_iter_{:05d}.pt".format(n))
            self._experiment.save_experiment(path=path)
        return log_dict

    def finalize(self, surrender: ExperimentSurrender = None, crash: bool = False):
        if self._experiment is not None:
            try:
                if (not crash) and isinstance(self._experiment, RLExperiment) and self._save_replay_buffer:
                    self._experiment.save_replay_buffer(path=self._log_path)
            except Exception as e:
                print(e)
        else:
            raise AssertionError

    def save_state(self, config: dict, rep: int, n: int) -> None:
        pass
