import argparse
import importlib.util

from lib.utils.exps_utils import init_exp


def main():
    args = parse_args()
    if args.temp_model_path:
        model_script = load_module(args.temp_model_path)
    else:
        model_script = load_module(args.model_path)

    cfg = init_exp(args)
    if cfg.test:
        model_script.test(cfg)
    else:
        model_script.train(cfg)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_path', type=str,
                        help='path to the model script')

    parser.add_argument('--exps-path', type=str,
                        default='',
                        help='path to the logs')

    parser.add_argument('--exp-name', type=str, default='')

    parser.add_argument('--dataset-path', type=str,
                        default='',
                        help='path to the dataset')

    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')

    parser.add_argument('--ngpus', type=int,
                        default=1,
                        help='number of GPUs')

    parser.add_argument('--gpus', type=str, default='', required=False)

    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')

    parser.add_argument('--no-states', action='store_true',
                        help='disable saving optimizer states')

    parser.add_argument('--resume-exp', type=str, default=None,
                        help='experiment to continue')

    parser.add_argument('--resume-prefix', type=str, default='latest',
                        help='prefix of checkpoint to continue training')

    parser.add_argument('--resume-no-states', action='store_true',
                        help='whether to load states')

    parser.add_argument('--temp-model-path', type=str, default='',
                        help='(for internal purposes)')

    parser.add_argument('--test', action='store_true',
                        help='run testing')

    parser.add_argument('--test-exp', type=str, default=None,
                        help='experiment to test')

    parser.add_argument('--test-prefix', type=str, default='latest',
                        help='prefix of checkpoint to test')

    parser.add_argument('--test-cmd', type=str, default='metrics',
                        help='test command')

    return parser.parse_args()


def load_module(script_path):
    spec = importlib.util.spec_from_file_location("model_script", script_path)
    model_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_script)

    return model_script


if __name__ == '__main__':
    main()
