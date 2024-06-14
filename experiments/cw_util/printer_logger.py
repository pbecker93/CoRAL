import logging
import sys
import os
import numpy as np

from cw2.cw_data.cw_logging import AbstractLogger


class _CWFormatter(logging.Formatter):

    def __init__(self):
        super(_CWFormatter, self).__init__()
        self.std_formatter = logging.Formatter('[%(name)s] %(message)s')
        self.red_formatter = logging.Formatter('[%(asctime)s] %(message)s')

    def format(self, record: logging.LogRecord):
        if record.levelno <= logging.ERROR:
            return self.std_formatter.format(record)
        else:
            return self.red_formatter.format(record)


class PrintLogger(AbstractLogger):

    def __init__(self, name: str, *args, **kwargs):
        super(PrintLogger, self).__init__(*args, **kwargs)
        self._name = name
        self._exclude_keys = []
        self._save_path = None
        self._logger = None

    def initialize(self, config: dict, rep: int, rep_log_path: str) -> None:
        formatter = _CWFormatter()

        self._exclude_keys = ["ts", "rep", "iter"]

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        sh.setLevel(logging.INFO)
        self._save_path = config["_rep_log_path"]
        fh = logging.FileHandler(os.path.join(self._save_path, "rkn.log"))
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)

        self._logger = logging.getLogger(name=self._name)
        self._logger.propagate = False
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(fh)
        self._logger.addHandler(sh)

        self._logger.info("------------------------------------------")
        self._logger.info("Starting Repetition {:03d}".format(rep))
        self._logger.info("------------------------------------------")

    def preprocess(self, log_dict: dict) -> None:
        save_dict = {}
        for k, v in log_dict.items():
            save_dict[k] = v.get_raw_dict()
            if isinstance(v, dict):
                for kk, vv in v.items():
                    self._logger.info("--- {} ---".format(kk))
                    line_list = str(vv).split("\n")[1:-1]
                    for line in line_list:
                        self._logger.info(line)
            else:
                self._logger.info("--- {} ---".format(k))
                line_list = str(v).split("\n")[1:-1]
                for line in line_list:
                    self._logger.info(line)
        np.savez(os.path.join(self._save_path, "config.npz"), **save_dict)

    def process(self, data) -> None:

        log_str = "Iteration  {:04d}: ".format(data["iter"])
        for k, v in data.items():
            if k not in self._exclude_keys:
                if isinstance(v, float):
                    log_str += "{}: {:.5f} ".format(k, v)
                elif isinstance(v, np.ndarray):
                    log_str += "{}: {} ".format(k,
                                                np.array2string(v, precision=5, max_line_width=int(1e300)))
                else:
                    log_str += "{}: {} ".format(k, str(v))
        self._logger.info(log_str)

    def finalize(self) -> None:
        for handler in self._logger.handlers:
            handler.close()
            self._logger.removeHandler(handler)

    def load(self):
        pass

    @property
    def raw_logger(self):
        return self._logger
