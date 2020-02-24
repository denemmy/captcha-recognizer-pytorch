import math
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
import functools


class ComposeScheduler(object):
    def __init__(self, segments, param_name='lr', init_values=None):
        self._optimizer = None
        self._segments = segments
        self.param_name = param_name
        self._init_values = init_values
        self._last_scale = 1.0
        self._last_segment_indx = 0
        self._prev_segments_length = 0

    def register_optimizer(self, optimizer):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self._optimizer = optimizer
        if self._init_values is None:
            self._init_values = self._get_param_values()

    def step(self, num_iters):
        scale = self.get_scale(num_iters)
        new_values = [scale * init_value for init_value in self._init_values]
        self._set_param_values(new_values)
        return new_values[0]

    def get_scale(self, num_iters):
        if num_iters < 0:
            raise RuntimeError

        if num_iters < self._prev_segments_length:
            self._prev_segments_length = 0
            self._last_segment_indx = 0
            self._last_scale = 1.0
            return self.get_scale(num_iters)

        if self._last_segment_indx >= len(self._segments):
            return self._last_scale

        last_segment, last_segment_length = self._segments[self._last_segment_indx]
        last_segment_iter = num_iters - self._prev_segments_length
        if last_segment_iter >= last_segment_length and last_segment_length > 0:
            self._prev_segments_length += last_segment_length
            self._last_segment_indx += 1
            self._last_scale *= last_segment.get_scale(last_segment_length)
            return self.get_scale(num_iters)

        return self._last_scale * last_segment.get_scale(last_segment_iter)

    def state_dict(self):
        return {
            '_init_values': self._init_values
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def _get_param_values(self):
        assert self._optimizer is not None
        values = []
        for group in self._optimizer.param_groups:
            if self.param_name in group:
                values.append(group[self.param_name])
        return values

    def _set_param_values(self, new_values):
        assert self._optimizer is not None

        index = 0
        for group in self._optimizer.param_groups:
            if self.param_name in group:
                group[self.param_name] = new_values[index]
                index += 1


class SchedulerSegment(object):
    def __init__(self, init_scale=1.0):
        self._init_scale = init_scale

    def get_scale(self, num_iter):
        return self._init_scale


class MultiplySegments(SchedulerSegment):
    def __init__(self, segments, **kwargs):
        super().__init__(**kwargs)
        self._segments = segments

    def get_scale(self, num_iter):
        return functools.reduce(lambda x, y: x * y.get_scale(num_iter),
                                self._segments, self._init_scale)


class ConstantSegment(SchedulerSegment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_scale(self, num_iter):
        return self._init_scale


class LinearSegment(SchedulerSegment):
    def __init__(self, target_scale, period, **kwargs):
        super().__init__(**kwargs)
        self._target_scale = target_scale
        self._period = period

    def get_scale(self, num_iter):
        if num_iter <= 0:
            return self._init_scale
        else:
            alpha = num_iter / self._period
            return (1 - alpha) * self._init_scale + alpha * self._target_scale


class ExponentialSegment(SchedulerSegment):
    def __init__(self, factor=0.5, factor_period=1000, min_scale=0.0, **kwargs):
        super().__init__(**kwargs)
        self._beta = math.pow(factor, 1.0 / factor_period)
        self._min_scale = min_scale

    def get_scale(self, num_iter):
        return self._init_scale * max(self._min_scale, math.pow(self._beta, num_iter))


class CosineSegment(SchedulerSegment):
    def __init__(self, period, min_scale=0.1, **kwargs):
        super().__init__(**kwargs)

        self._min_scale = min_scale
        self._period = period

    def get_scale(self, num_iter):
        t = math.pi * num_iter / self._period
        v = (math.cos(t) + 1) / 2.0
        return self._init_scale * (self._min_scale + v * (1 - self._min_scale))


class ExpSineSegment(SchedulerSegment):
    def __init__(self, period, amplitude=2, shift=False, **kwargs):
        super().__init__(**kwargs)

        self._amplitude = amplitude
        self._period = period
        self._shift = shift

    def get_scale(self, num_iter):
        t = 2 * math.pi * (num_iter / self._period)
        if self._shift:
            t += math.pi
        return self._init_scale * math.pow(self._amplitude, math.sin(t))


class LinearDecayLR(_LRScheduler):

    def __init__(self, optimizer, total_iters=None, decay_after=0, target_lr_mult=0, last_epoch=-1):
        assert total_iters is not None
        self.decay_after = decay_after
        self.total_iters = total_iters
        self.target_lr_mult = target_lr_mult
        super(LinearDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.decay_after:
            alpha = (self.last_epoch - self.decay_after) / (self.total_iters - self.decay_after)
            gamma = 1.0 + (self.target_lr_mult - 1.0) * alpha
        else:
            gamma = 1.0

        return [group['initial_lr'] * gamma for group in self.optimizer.param_groups]
