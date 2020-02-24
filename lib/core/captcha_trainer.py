import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
from lib.utils.log import logger
import numpy as np
import cv2
import copy
from lib.data.iterators import get_infinity_iterator
from torch.utils.tensorboard import SummaryWriter
from .event_loop import EventLoop
from lib.metrics.common import CaptchaAccuracy
from tqdm import tqdm
from lib.utils.visualization import labels_to_imgs


class CaptchaTrainer:
    def __init__(self, cfg, net, optim_params, train_dataset,
                 val_dataset, image_normalization=None,
                 total_kimgs=1000, batch_size=64,
                 log_period_kimgs=10, log_images_period_kimgs=10, n_display=32, eval_period=10,
                 last_checkpoint_period_kimgs=50, beta_avg=0, checkpoint_period_kimgs=-1):

        self.n_display = n_display
        self.batch_size = batch_size
        self.total_kimgs = total_kimgs
        self.cfg = cfg
        self.beta_avg = beta_avg
        self.image_normalization = image_normalization

        self.eval_captch_accuracy = 0.
        self.eval_symbol_accuracy = 0.
        self.eval_batch = None
        self.train_batch = None

        self.eval_period = eval_period
        self.log_period_kimgs = log_period_kimgs
        self.log_images_period_kimgs = log_images_period_kimgs
        self.last_checkpoint_period_kimgs = last_checkpoint_period_kimgs
        self.checkpoint_period_kimgs = checkpoint_period_kimgs

        self.sw = SummaryWriter(log_dir=str(cfg.logs_path))
        self.event_loop = EventLoop(console_logger=logger, tb_writer=self.sw)

        gpu_ids = cfg.gpu_ids
        ngpu = len(gpu_ids)

        device = torch.device(gpu_ids[0] if (torch.cuda.is_available() and ngpu > 0) else 'cpu')
        self.device = device
        self.use_avg = True if beta_avg > 0 else False
        self.avg_inited = False

        self.net = net
        if self.use_avg:
            netA = copy.deepcopy(self.net) if self.use_avg else None
            netA = netA.to(device)
            netA.eval()
            self.netA = netA
        else:
            self.netA = None

        self.net = self.net.to(device)
        self.net.train()

        self.ce_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.optim_params = optim_params
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.init_optimizer()
        self.init_dataloaders()

        self.display_dataset_examples(self.train_dataset, prefix='train')
        if self.val_dataset is not None:
            self.display_dataset_examples(self.val_dataset, prefix='val')

        if cfg.resume_exp:
            self.load()

        self.print_network_params(self.net, 'net', verbose=True)

        if (device.type == 'cuda') and (ngpu > 1):
            self.net = nn.DataParallel(self.net, gpu_ids)

        self.register_events()

    def register_events(self):

        self.event_loop.register_metric('ce_loss', console_name='CELoss', console_period=self.log_period_kimgs,
                                        tb_name='ce_loss', tb_period=1, counter='ma_10')

        if self.val_dataset is not None:
            self.event_loop.register_metric_event(self.evaluate, metric_name='captcha_accuracy',
                                                  period=self.eval_period, tb_global_step='n_periods',
                                                  tb_name='val/captcha_accuracy')
            self.event_loop.register_metric_event(self.get_last_evaluate_symbol_accuracy,
                                                  metric_name='symbol_accuracy', period=self.eval_period,
                                                  tb_global_step='n_periods', tb_name='val/symbol_accuracy')

        for scheduler in self.optimizer_schedulers:
            logger.info('register optim')
            scheduler.register_optimizer(self.optimizer)
            self.event_loop.register_metric_event(scheduler.step, func_inputs=('f_periods',),
                                                  metric_name=f'net_{scheduler.param_name}', console_format='{}',
                                                  period=0, console_period=-1, tb_period=2, tb_global_step='n_periods',
                                                  tb_name=f'TrainStates/net_{scheduler.param_name}')

        self.event_loop.register_event(self.images_log, period=self.log_images_period_kimgs, func_inputs=('n_periods',))
        self.event_loop.register_event(self.save, period=self.checkpoint_period_kimgs, func_inputs=('n_periods',))
        self.event_loop.register_event(self.save, period=self.last_checkpoint_period_kimgs)

    def init_optimizer(self):

        optim_params = self.optim_params
        opt = optim_params.opt
        net_params = self.net.parameters()
        self.optimizer = opt(net_params, **optim_params.params)
        self.optimizer_schedulers = getattr(optim_params, 'schedulers', [])

    def init_dataloaders(self):

        train_data = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True,
                                                  shuffle=True, num_workers=self.cfg.workers)
        self.train_data = get_infinity_iterator(train_data)

        if self.val_dataset is not None:
            self.val_data = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size,
                                                        shuffle=False, num_workers=self.cfg.workers)
        else:
            self.val_data = None

    def train(self):
        logger.info('starting training loop...')

        while self.event_loop.n_periods < self.total_kimgs:
            self.optim_step()
            if self.use_avg:
                self.update_average()
            self.event_loop.step(self.batch_size)

        self.save(prefix='latest')
        logger.info('all done.')

    def optim_step(self):
        batch = next(self.train_data)

        imgs = batch['img'].to(self.device)
        labels = batch['label'].to(self.device)

        self.optimizer.zero_grad()

        outputs = self.net(imgs)

        labels = labels.view((labels.size(0), -1))
        outputs = outputs.view((outputs.size(0), outputs.size(1), -1))

        ce_loss = self.ce_criterion(outputs, labels)

        ce_loss.backward()
        self.optimizer.step()

        self.event_loop.add_metric_value('ce_loss', ce_loss.item())

    def evaluate(self):

        val_iter = iter(self.val_data)

        net = self.netA if self.use_avg else self.net
        is_training = net.training
        net.eval()

        metric = CaptchaAccuracy()

        with torch.no_grad():
            for data in val_iter:
                inputs = data['img']
                labels = data['label']
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = net(inputs)
                outputs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                metric.update(labels, preds)

        if is_training:
            net.train()

        symbol_accuracy, captcha_accuracy = metric.get()
        self.eval_symbol_accuracy = symbol_accuracy
        self.eval_captcha_accuracy = captcha_accuracy
        return captcha_accuracy

    def get_last_evaluate_symbol_accuracy(self):
        return self.eval_symbol_accuracy

    def get_last_evaluate_captcha_accuracy(self):
        return self.eval_captcha_accuracy

    def save(self, prefix='latest'):
        logger.info('saving checkpoint')
        checkpoint_path = self.cfg.checkpoints_path / f'{prefix}_checkpoint.tar'

        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module

        checkpoint_states = {
            'net': net.state_dict()
        }
        if self.use_avg:
            netA = self.netA
            if isinstance(netA, torch.nn.DataParallel):
                netA = netA.module
            checkpoint_states['netA'] = netA.state_dict(),

        if not self.cfg.no_states:

            checkpoint_states['opt_states'] = self.optimizer.state_dict()
            checkpoint_states['event_loop_states'] = self.event_loop.get_states()

            if len(self.optimizer_schedulers) > 0:
                checkpoint_states['schedulers'] = [sched.state_dict() for sched in self.optimizer_schedulers]

        torch.save(checkpoint_states, checkpoint_path)

    def load(self):
        prefix = self.cfg.resume_prefix
        checkpoint_path = self.cfg.checkpoints_path / f'{prefix}_checkpoint.tar'
        checkpoint_states = torch.load(checkpoint_path)
        logger.info(f'loading checkpoint "{checkpoint_path}"')

        self.net.load_state_dict(checkpoint_states['net'], strict=True)
        if self.use_avg and 'netA' in checkpoint_states:
            self.netA.load_state_dict(checkpoint_states['netA'], strict=True)
            self.avg_inited = True

        if not self.cfg.resume_no_states:

            self.event_loop.set_states(checkpoint_states['event_loop_states'])
            self.optimizer.load_state_dict(checkpoint_states['opt_states'])

            if 'schedulers' in checkpoint_states:
                assert len(self.optimizer_schedulers) == len(checkpoint_states['schedulers'])
                for sched, state in zip(self.optimizer_schedulers, checkpoint_states['schedulers']):
                    sched.load_state_dict(state)

    def prepare_vis_pairs(self, imgs, labels, ignore=None):
        if self.image_normalization is not None:
            imgs = 255. * self.image_normalization(imgs, inverse=True)

        n_samples = imgs.size(0)
        sqrt_val = max(1, np.sqrt(n_samples))
        nrow = 2 ** int(np.floor(np.log2(sqrt_val)))

        if ignore is not None:
            labels[ignore] = -1
        labels = labels.cpu().numpy()
        resolve_texts = [self.decode(l) for l in labels]
        masks = labels_to_imgs(imgs.shape, resolve_texts).to(imgs.device)
        imgs = torch.cat([imgs, masks], dim=3)

        imgs = np.transpose(vutils.make_grid(imgs, nrow=nrow,
                                             padding=2, normalize=False).cpu().numpy(), (1, 2, 0))
        imgs = imgs[:, :, ::-1].astype(np.uint8)
        return imgs

    def decode(self, labels):
        resolve_text = ''
        for l in labels:
            symbol = chr(ord('a') + l) if l >= 0 else ' '
            resolve_text += symbol
        return resolve_text

    def run_batches(self, large_batch):

        net = self.netA if self.use_avg else self.net
        is_training = net.training
        net.eval()

        n_samples = large_batch.shape[0]

        with torch.no_grad():
            if n_samples < self.batch_size:
                batch = large_batch.to(self.device)
                output = net(batch)
                output = output.detach().cpu()
            else:
                n_batches = n_samples // self.batch_size
                n_batches += 1 if n_samples % self.batch_size > 0 else 0

                output = None
                for n_batch in range(n_batches):
                    st = n_batch * self.batch_size
                    en = (n_batch + 1) * self.batch_size
                    en = min(en, n_samples)
                    batch_s = large_batch[st:en].to(self.device)
                    output_s = net(batch_s)
                    output_s = output_s.detach().cpu()
                    output = output_s if output is None else torch.cat([output, output_s], dim=0)

        if is_training:
            net.train()

        pred = torch.argmax(output, dim=1)

        return pred

    def images_log(self, n_periods):

        if self.eval_batch is None and self.val_dataset is not None:
            n_display = min(len(self.val_dataset), self.n_display)
            idx = np.random.choice(len(self.val_dataset), n_display, replace=False)
            self.eval_batch = self.prepare_batch_from_idx(idx, self.val_dataset)

            imgs = self.prepare_vis_pairs(self.eval_batch['img'],
                                          self.eval_batch['label'])
            path_to_save = self.cfg.vis_path / f'progress_val_gt.jpg'
            cv2.imwrite(str(path_to_save), imgs)

        if self.train_batch is None:
            n_display = min(len(self.train_dataset), self.n_display)
            idx = np.random.choice(len(self.train_dataset), n_display, replace=False)
            self.train_batch = self.prepare_batch_from_idx(idx, self.train_dataset)

            imgs = self.prepare_vis_pairs(self.train_batch['img'],
                                          self.train_batch['label'])
            path_to_save = self.cfg.vis_path / f'progress_train_gt.jpg'
            cv2.imwrite(str(path_to_save), imgs)

        if self.eval_batch is not None:
            pred_labels = self.run_batches(self.eval_batch['img'])
            ignore = self.eval_batch['label'] == -1
            imgs = self.prepare_vis_pairs(self.eval_batch['img'], pred_labels, ignore=ignore)
            path_to_save = self.cfg.vis_path / f'progress_val_kimg_{n_periods:06d}.jpg'
            cv2.imwrite(str(path_to_save), imgs)

        pred_labels = self.run_batches(self.train_batch['img'])
        ignore = self.train_batch['label'] == -1
        imgs = self.prepare_vis_pairs(self.train_batch['img'], pred_labels, ignore=ignore)
        path_to_save = self.cfg.vis_path / f'progress_train_kimg_{n_periods:06d}.jpg'
        cv2.imwrite(str(path_to_save), imgs)

    def prepare_batch_from_idx(self, idx, dataset):

        batch = None
        sel_keys = None

        for i in range(len(idx)):
            data_i = dataset[idx[i]]

            if sel_keys is None:
                sel_keys = []
                for k in data_i:
                    if isinstance(data_i[k], torch.Tensor):
                        sel_keys.append(k)

            if batch is None:
                batch = {}
                for k in sel_keys:
                    batch[k] = torch.zeros((len(idx),) + data_i[k].shape, dtype=data_i[k].dtype)
            for k in sel_keys:
                batch[k][i] = data_i[k]

        return batch

    def display_dataset_examples(self, dataset, prefix=''):

        display_real_idx = np.random.choice(len(dataset), self.n_display, replace=False)

        display_data = self.prepare_batch_from_idx(display_real_idx, dataset)
        imgs = self.prepare_vis_pairs(display_data['img'], display_data['label'])

        prefix = f'{prefix}_' if prefix else ''
        path_to_save = self.cfg.vis_path / f'{prefix}image.jpg'
        cv2.imwrite(str(path_to_save), imgs)

    def update_average(self):

        initialize = not self.avg_inited

        net = self.net
        netA = self.netA
        beta = self.beta_avg

        state_dict = net.state_dict()
        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict}
        state_dictA = netA.state_dict()

        always_copy = ['.running_mean', '.running_var', '.num_batches_tracked']

        with torch.no_grad():
            for name in state_dict:
                param = state_dict[name]
                paramA = state_dictA[name]

                do_copy = False
                for copy_name in always_copy:
                    if name.endswith(copy_name):
                        do_copy = True
                        break

                if do_copy or initialize:
                    state_dictA[name].data.copy_(param.data)
                else:
                    state_dictA[name].data.copy_(beta * paramA.data + (1 - beta) * param.data)

        netA.load_state_dict(state_dictA)
        self.avg_inited = True

    def print_network_params(self, net, name='', verbose=False):
        logger.info('-----------------------------------------------')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        if verbose:
            logger.info(net)
        logger.info('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        logger.info('-----------------------------------------------')


class CaptchaTester:
    def __init__(self, cfg, net, test_dataset=None, image_normalization=None, batch_size=64, use_avg=False,
                 n_display=None):

        self.batch_size = batch_size
        self.cfg = cfg
        self.use_avg = use_avg
        self.num_classes = test_dataset.num_classes()
        self.n_display = n_display
        self.image_normalization = image_normalization

        gpu_ids = cfg.gpu_ids
        ngpu = len(gpu_ids)

        device = torch.device(gpu_ids[0] if (torch.cuda.is_available() and ngpu > 0) else 'cpu')
        self.device = device
        self.net = net.to(device)
        self.net.eval()

        self.test_dataset = test_dataset
        self.init_dataloaders()

        self.load()

        if (device.type == 'cuda') and (ngpu > 1):
            self.net = nn.DataParallel(self.net, gpu_ids)

    def init_dataloaders(self):
        self.test_data = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, drop_last=False,
                                                     shuffle=True, num_workers=self.cfg.workers)

    def evaluate(self):

        metric = CaptchaAccuracy()
        test_iter = iter(self.test_data)

        with torch.no_grad():
            with tqdm(total=len(self.test_dataset)) as pb:
                for data in test_iter:
                    inputs = data['img']
                    labels = data['label']
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.net(inputs)
                    outputs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs)
                    metric.update(labels, preds)
                    pb.update(inputs.size(0))
                    symbol_accuracy, captcha_accuracy = metric.get()
                    pb.set_description(f'captcha_accuracy: {captcha_accuracy:.3f}, symbol_accuracy: {symbol_accuracy:.3f}')

        symbol_accuracy, captcha_accuracy = metric.get()
        return {'captcha_accuracy': captcha_accuracy, 'symbol_accuracy': symbol_accuracy}

    def load(self):
        prefix = self.cfg.test_prefix
        checkpoint_path = self.cfg.checkpoints_path / f'{prefix}_checkpoint.tar'
        checkpoint_states = torch.load(checkpoint_path, map_location=self.device)
        logger.info(f'loading checkpoint "{checkpoint_path}"')

        if self.use_avg:
            self.net.load_state_dict(checkpoint_states['netA'], strict=True)
        else:
            self.net.load_state_dict(checkpoint_states['net'], strict=True)

    def display_dataset_examples(self, prefix=''):

        dataset = self.test_dataset

        display_real_idx = np.random.choice(len(dataset), self.n_display, replace=False)

        display_data = None
        for i in range(self.n_display):
            data_i = dataset[display_real_idx[i]]
            if display_data is None:
                display_data = {}
                for k in data_i:
                    display_data[k] = torch.zeros((self.n_display,) + data_i[k].shape, dtype=data_i[k].dtype)
            for k in data_i:
                display_data[k][i] = data_i[k]

        data_to_display = display_data['img']
        if self.image_normalization is not None:
            data_to_display = self.image_normalization(data_to_display, inverse=True)

        n_samples = data_to_display.size(0)
        sqrt_val = max(1, np.sqrt(n_samples))
        nrow = 2 ** int(np.floor(np.log2(sqrt_val)))

        imgs = np.transpose(vutils.make_grid(data_to_display, nrow=nrow,
                                             padding=2, normalize=False).cpu().numpy(), (1, 2, 0))
        imgs = (255 * imgs[:,:,::-1]).astype(np.uint8)

        path_to_save = self.cfg.vis_path / f'{prefix}test_examples.jpg'
        cv2.imwrite(str(path_to_save), imgs)
