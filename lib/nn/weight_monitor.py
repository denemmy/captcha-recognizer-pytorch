import torch


class WeightMonitor(object):
    def __init__(self):
        self.event_loop = None
        self.registered_weights = set()
        self.weights_params = dict()

    def log_weight(self, weight_name, weight):
        if self.event_loop is None:
            return

        if '/' not in weight_name:
            tb_weight_name = '/' + weight_name
        else:
            tb_weight_name = weight_name

        if weight_name not in self.registered_weights:
            wparams = self.weights_params[weight_name]
            self.event_loop.register_metric(f'{weight_name}_norm', console_period=-1,
                                            tb_name=f'WeightStats{tb_weight_name}_norm',
                                            tb_period=wparams['period'], counter=None)
            self.event_loop.register_metric(f'{weight_name}_std', console_period=-1,
                                            tb_name=f'WeightStats{tb_weight_name}_std',
                                            tb_period=wparams['period'], counter=None)
            self.registered_weights.add(weight_name)

        with torch.no_grad():
            self.event_loop.add_metric_value(f'{weight_name}_std', torch.std(weight).item())
            self.event_loop.add_metric_value(f'{weight_name}_norm', torch.norm(weight, p=2).item())

    def register_weight(self, weight_name, period):
        self.weights_params[weight_name] = {'period': period}

    def set_event_loop(self, event_loop):
        self.event_loop = event_loop


def monitor_weight(net, name, period, weight_name='weight'):
    def forward_pre_hook(module, input):
        weight = getattr(module, weight_name)
        wm.log_weight(name, weight)

    wm.register_weight(name, period)
    net.register_forward_pre_hook(forward_pre_hook)

    return net


wm = WeightMonitor()