import os
import sys
import yaml
import shutil
import pickle
import pprint
from pathlib import Path
from datetime import datetime
from easydict import EasyDict as edict
from .log import *


def init_exp(args):
    model_path = Path(args.model_path)
    model_name = model_path.stem

    cfg = load_config(model_path)
    update_config(cfg, args)

    experiments_path = Path(cfg.exps_path)

    exp_parent_path = experiments_path / model_path.parent.stem / model_name
    if not cfg.test:
        exp_parent_path.mkdir(parents=True, exist_ok=True)

    if cfg.test:
        exp_path = find_resume_exp(exp_parent_path, cfg.test_exp)
    elif cfg.resume_exp:
        exp_path = find_resume_exp(exp_parent_path, cfg.resume_exp)
    else:
        cleanup_experiments(cfg, exp_parent_path)

        last_exp_indx = find_last_exp_indx(exp_parent_path)
        exp_name = f'{last_exp_indx:03d}'
        if cfg.exp_name:
            exp_name += '_' + cfg.exp_name
        exp_path = exp_parent_path / exp_name
        exp_path.mkdir(parents=True)

    cfg.exp_path = exp_path
    cfg.checkpoints_path = exp_path / 'checkpoints'
    cfg.vis_path = exp_path / 'vis'
    cfg.logs_path = exp_path / 'logs'
    cfg.graphs_path = exp_path / 'graphs'

    cfg.logs_path.mkdir(exist_ok=True)
    if not cfg.test:
        cfg.checkpoints_path.mkdir(exist_ok=True)
        cfg.vis_path.mkdir(exist_ok=True)
        cfg.graphs_path.mkdir(exist_ok=True)

        if args.temp_model_path:
            shutil.copy(args.temp_model_path, exp_path / model_path.name)
            os.remove(args.temp_model_path)
        else:
            shutil.copy(model_path, exp_path / model_path.name)

    if cfg.gpus != '':
        gpu_ids = [int(id) for id in cfg.gpus.split(',')]
        cfg.gpu_ids = gpu_ids
    else:
        cfg.gpu_ids = list(range(cfg.ngpus))

    add_logging(cfg.logs_path, is_test=cfg.test)

    logger.info(f'Number of GPUs: {len(cfg.gpu_ids)}')

    logger.info('Run experiment with config:')
    logger.info(pprint.pformat(cfg, indent=4))

    return cfg


def load_config(model_path):
    model_name = model_path.stem
    config_path = model_path.parent / (model_name + '.yml')

    if config_path.exists():
        cfg = load_config_file(config_path)
    else:
        cfg = dict()

    cwd = Path.cwd()
    config_parent = config_path.parent.absolute()
    while len(config_parent.parents) > 0:
        config_path = config_parent / 'config.yml'

        if config_path.exists():
            local_config = load_config_file(config_path, model_name=model_name)
            cfg.update({k: v for k, v in local_config.items() if k not in cfg})

        if config_parent.absolute() == cwd:
            break
        config_parent = config_parent.parent

    return edict(cfg)


def load_config_file(config_path, model_name=None):
    with open(config_path, 'r') as f:
        cfg = yaml.load(f)

    if 'SUBCONFIGS' in cfg:
        if model_name is not None and model_name in cfg['SUBCONFIGS']:
            cfg.update(cfg['SUBCONFIGS'][model_name])
        del cfg['SUBCONFIGS']

    cfg = {k.lower(): v for k, v in cfg.items()}

    return cfg


def update_config(cfg, args):
    for param_name, value in vars(args).items():
        if param_name.lower() in cfg or param_name.upper() in cfg:
            continue
        cfg[param_name.lower()] = value


def find_resume_exp(exp_parent_path, exp_pattern):
    candidates = sorted(exp_parent_path.glob(f'{exp_pattern}*'))
    if len(candidates) == 0:
        print(f'No experiments could be found that satisfies the pattern = "*{exp_pattern}"')
        sys.exit(1)
    elif len(candidates) > 1:
        print('More than one experiment found:')
        for x in candidates:
            print(x)
        sys.exit(1)
    else:
        exp_path = candidates[0]
        # print(f'Continue with experiment "{exp_path}"')

    return exp_path


def find_last_exp_indx(exp_parent_path):
    indx = 0
    for x in exp_parent_path.iterdir():
        if not x.is_dir():
            continue

        exp_name = x.stem
        if exp_name[:3].isnumeric():
            indx = max(indx, int(exp_name[:3]) + 1)

    return indx


def add_logging(logs_path, is_test):
    prefix = 'test_' if is_test else 'train_'
    log_name = prefix + datetime.strftime(datetime.today(), '%Y-%m-%d_%H-%M-%S') + '.log'
    stdout_log_path = logs_path / log_name

    fh = logging.FileHandler(str(stdout_log_path))
    formatter = logging.Formatter(fmt='(%(levelname)s) %(asctime)s: %(message)s',
                                  datefmt=LOGGER_DATEFMT)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def cleanup_experiments(cfg, exps_path):
    trash_exps_path = exps_path / 'trash'

    for exp_path in exps_path.iterdir():
        if not exp_path.is_dir() or exp_path.name == 'trash':
            continue

        exp_duration = get_exp_duration(exp_path, cfg.cleanup_block_timeout_sec)
        if exp_duration < cfg.cleanup_min_duration:
            print(f'Experiment "{exp_path.name}" (duration={exp_duration:.1f} s) is archived')
            move_exp_to_trash(exp_path, trash_exps_path)

    if trash_exps_path.exists():
        cleanup_trash_exps(cfg, trash_exps_path)


def get_exp_duration(exp_path, block_timeout_sec):
    states_path = exp_path / '.exp_states'
    if states_path.exists():
        with open(states_path, 'rb') as f:
            min_start_time, max_end_time = pickle.load(f)
    else:
        min_start_time = None
        max_end_time = None

        for log_path in (exp_path / 'logs').glob('*.log'):
            start_time, end_time = get_log_start_end_time(log_path)
            if min_start_time is None or start_time < min_start_time:
                min_start_time = start_time

            if max_end_time is None or end_time > max_end_time:
                max_end_time = end_time

        if max_end_time is None:
            return 0

        delta = (datetime.today() - max_end_time).total_seconds()
        if delta < block_timeout_sec:
            return 10 ** 6

        with open(states_path, 'wb') as f:
            pickle.dump((min_start_time, max_end_time), f)

    delta = max_end_time - min_start_time
    return delta.total_seconds()


def move_exp_to_trash(exp_path, trash_exps_path):
    exp_new_name = exp_path.stem + '_' + datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M-%S')
    trash_exps_path.mkdir(parents=True, exist_ok=True)
    shutil.move(exp_path, trash_exps_path / exp_new_name)


def cleanup_trash_exps(cfg, trash_exps_path):
    if not trash_exps_path.exists():
        return

    alive = 0
    for x in trash_exps_path.iterdir():
        if not x.is_dir():
            continue

        move_date = x.stem.split('_')[-1]
        move_date = datetime.strptime(move_date, '%Y-%m-%d-%H-%M-%S')
        delta = (datetime.today() - move_date).total_seconds() / 3600
        if delta >= cfg.cleanup_remove_period:
            shutil.rmtree(x)
        else:
            alive += 1

    if alive == 0:
        shutil.rmtree(trash_exps_path)


def get_log_start_end_time(log_path):
    def get_date_from_log_line(log_line):
        assert log_line[:7] == '(INFO) '

        result = log_line[7:7 + 19]
        result = datetime.strptime(result, '%Y-%m-%d %H:%M:%S')
        return result

    with open(log_path, 'r') as f:
        lines = f.readlines()

    lines = [x for x in lines if x.startswith('(INFO)')]
    start_time = get_date_from_log_line(lines[0])
    end_time = get_date_from_log_line(lines[-1])

    return start_time, end_time
