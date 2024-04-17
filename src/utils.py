import collections
import numpy as np
import torch
import yaml, os
from torch.utils.data.sampler import SubsetRandomSampler
from functools import partial
from tycho_env.utils import colors, print_and_cr

cameras_operation = {
    #'435': [None, None, None, None, None, None, None],
    #'415_1': [None, None, None, None, None, None, None],
    '435': [480-202-188, 188, 138, 300, None, None, None],
    '415_1': [480-10-314, 314, 139, 314, None, None, None],
    '415_2': [480-70-274, 70, 273, 93, None, None, None],
    'kinect': [20, 700, 120, 700, None, None, None],
    'note': ['crop_y','crop_h','crop_x','crop_w',
             'resize_w', 'resize_h','resize_interpolation',
             'leave NONE if you dont want crop or resize']
}

def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_yaml_config(model_folder=None, yaml_path=None, overrides=None, test=False):
  yaml_path = yaml_path or os.path.join(model_folder, 'config.yaml')
  with open(yaml_path) as f:
    config = yaml.load(f)
  if overrides:
    config.update(overrides)
    config = parse_overrides(config, overrides)
  if config['data']['cameras'] == 0 or \
    '0' in config['data']['cameras'] or len(config['data']['cameras']) == 0:
    print_and_cr(colors.fg.yellow + "Not using any cameras" + colors.reset)
    config['data']['cameras'] = []
    config['model']['im_out_size'] = 0
  if test:
    config['training']['data_noise_cov'] = None
  return Namespace(config)


def update_loss_fn_and_reduction(config, agent):
    if config.training.loss_fn == 'mse':
        from ChopAgentBC import loss_fn_mse
        agent.loss_fn = loss_fn_mse
    elif config.training.loss_fn == 'mse_alt':
        from ChopAgentBC import loss_fn_mse_alt
        agent.loss_fn = loss_fn_mse_alt
    elif config.training.loss_fn == 'quat':
        from ChopAgentBC import loss_fn_considering_quaternion
        agent.loss_fn = loss_fn_considering_quaternion
    elif config.training.loss_fn == 'default':
        from ChopAgentBC import list_loss_reduction
        agent.loss_reduction = partial(list_loss_reduction, config.training.loss_weights)
    else:
        assert False, "wrong loss fn for BC"

def init_agent_and_load(config, device, model_path=None, preload_dataset=None):
    if "KNN" in config.model_type:
        from ChopAgentKNN import e_to_negative_x, uniform_weights
        knn_weight_fn_dict = {
            'e_to_negative_x': e_to_negative_x,
            'uniform_weights': uniform_weights
        }
    # If given model_path, will try to load a pretrained model
    if config.model_type == 'KNN':
        dataset = preload_dataset or \
                load_datasets(config, 'cpu', to_tensor=False)
        from ChopAgentKNN import ChopAgentKNN
        agent = ChopAgentKNN(
            k=config.model.knn_k, weight_fn=knn_weight_fn_dict[config.model.weight_fn],
            X=np.copy(dataset.cache_choppose),
            Y=np.copy(dataset.cache_target),
            normalization=config.model.normalization,
            norm_weights=config.model.norm_weights,
            ngram=config.data.input_ngram)

    elif config.model_type == 'BC':
        from ChopAgentBC import init_models_from_config, ChopAgentBC, loss_reduction
        models = init_models_from_config(config, device)
        agent = ChopAgentBC(models,
                  learning_rate=config.training.learning_rate,
                  loss_reduction=partial(loss_reduction, config.training.loss_weights),
                  data_noise_cov=config.training.data_noise_cov,
                  noise_portion=config.training.data_noise_portion,
                  )
        update_loss_fn_and_reduction(config, agent)

    elif config.model_type == 'BCxKNN':
        from ChopAgentBC import loss_reduction
        from ChopAgentBCxKNN import init_models_from_config, ChopAgentBCxKNN
        from ChopAgentKNN import ChopAgentKNN
        models = init_models_from_config(config, device)
        knn_dataset = load_datasets(config.knn, 'cpu', to_tensor=False)
        knn = ChopAgentKNN(k=config.knn.k,
                           weight_fn=knn_weight_fn_dict[config.knn.weight_fn],
                           X=knn_dataset.cache_choppose, Y=knn_dataset.cache_target,
                           normalization=config.knn.normalization,
                           norm_weights=config.knn.norm_weights,
                           ngram=config.knn.data.input_ngram)
        agent = ChopAgentBCxKNN(models, knn,
                  learning_rate=config.training.learning_rate,
                  loss_reduction=partial(loss_reduction, config.training.loss_weights),
                  knn_output=config.knn.outputs,
                  )
        update_loss_fn_and_reduction(config, agent)

    elif config.model_type == 'KNNxBC':
        from ChopAgentBC import loss_reduction
        from ChopAgentKNN import ChopAgentKNN
        from ChopAgentKNNxBC import init_models_from_config, ChopAgentKNNxBC
        models = init_models_from_config(config, device)
        knn_dataset = load_datasets(config.knn, 'cpu', to_tensor=False)
        knn = ChopAgentKNN(k=config.knn.k,
                           weight_fn=knn_weight_fn_dict[config.knn.weight_fn],
                           X=knn_dataset.cache_choppose, Y=knn_dataset.cache_target,
                           normalization=config.knn.normalization,
                           norm_weights=config.knn.norm_weights,
                           ngram=config.knn.data.input_ngram)
        expert_dataset = load_datasets(config.expert, device)
        agent = ChopAgentKNNxBC(models, knn,
                  expert_dataset,
                  learning_rate=config.training.learning_rate,
                  loss_reduction=partial(loss_reduction, config.training.loss_weights),
                  data_noise_cov=config.training.data_noise_cov,
                  noise_portion=config.training.data_noise_portion,
                  )
        update_loss_fn_and_reduction(config, agent)

    elif config.model_type == 'DaD': # for now copy from BC
        from ChopAgentBC import init_models_from_config, ChopAgentBC, loss_reduction
        models = init_models_from_config(config, device)
        agent = ChopAgentBC(models,
                  learning_rate=config.training.learning_rate,
                  loss_reduction=partial(loss_reduction, config.training.loss_weights),
                  data_noise_cov=config.training.data_noise_cov,
                  noise_portion=config.training.data_noise_portion,
                  )
        update_loss_fn_and_reduction(config, agent)

    elif config.model_type == 'VAE':
        from ChopAgentVAE import init_models_from_config, ChopAgentVAE
        models = init_models_from_config(config, device)
        agent = ChopAgentVAE(models,
                  learning_rate=config.training.learning_rate,
                  loss_reduction=partial(custom_weight_loss_reduction, config.training.loss_weights),
                  kl_beta=config.training.kl_beta
                  )

    elif config.model_type == 'MJRL':
        from MJRLAgent import init_agent_from_config
        agent = init_agent_from_config(config, device)

    elif config.model_type == 'D3RLPY':
        from D3Agent import init_agent_from_config
        agent = init_agent_from_config(config, device)

    elif config.model_type == 'Diffusion':
        from DiffusionAgent import init_agent_from_config
        agent = init_agent_from_config(config, device)

    else:
        assert False, 'Unrecognized model type'

    if model_path is not None:
        try:
            print(f"Loading from: {model_path}")
            agent.load(model_path, device)
            print(colors.fg.green + 'Load previous model from{}\n'.format(model_path) + colors.reset)
        except Exception as e:
            print(colors.fg.red + colors.bold + 'Did not find previous model to load' + colors.reset)
            print(e)
            pass
        print(colors.reset)
    return agent

def generate_camera_transforms(config):
    from torchvision import transforms
    return transforms.Compose([
                    transforms.Resize((config.model.im_h, config.model.im_w)),
                    transforms.ToTensor(),
                    # transforms.Normalize([0,0,0], [255,255,255], inplace=True),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True),
                    ])

def convertDictToTensor(transform_img, sample, device='cpu'):
    my_tensor = {}
    for _k, _v in sample.items():
        if _k.startswith('camera'):
            my_tensor[_k] = transform_img(_v).float()
        else:
            my_tensor[_k] = torch.from_numpy(_v).float()
        my_tensor[_k] = my_tensor[_k].to(device)
    return my_tensor


def load_datasets(config, device, to_tensor=True, ENVVAR='CHOPDATA'):
    from ChopDataset import ChopDataset
    from ChopDataset_with_velocity import ChopDataset_with_velocity
    from torch.utils.data.dataset import ConcatDataset
    import os
    print("***************************\n", config)
    data_prefix = os.environ.get(ENVVAR) if config.data.folders[0][0] != '/' else ''
    datasets = [ChopDataset_with_velocity(
            os.path.join(data_prefix, _data),
            cameras=config.data.cameras,
            transform_img=generate_camera_transforms(config),
            input_ngram=config.data.input_ngram,
            output_ngram=config.data.output_ngram,
            input_command=config.data.input_command,
            input_goal=config.data.input_goal,
            input_last_torque=config.data.input_last_torque,
            returnTrajectory=False,
            device=device,
            selTraj=config.data.selTraj,
            trajNum=config.data.data_size,
            shuffle=config.data.shuffle_dataset,
            to_tensor=to_tensor,
            window=None,
            discount_factor=config.data.discount_factor or 1.0,
            obj_centric=config.data.obj_centric or False,
            obj_vel=config.data.obj_velocity or False,
            ) for _data in config.data.folders]
    if len(config.data.folders) == 1:
        return datasets[0]
    return ConcatDataset(datasets)

def split_dataset(dataset, config, dataloader_kwargs):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(config.training.validation_split * dataset_size))
    if config.training.shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset,
                   batch_size=config.training.batch_size,
                   sampler=train_sampler,
                   collate_fn=None,
                   **dataloader_kwargs)
    valid_loader = torch.utils.data.DataLoader(dataset,
                   batch_size=config.training.batch_size,
                   sampler=valid_sampler,
                   collate_fn=None,
                   **dataloader_kwargs)
    return train_loader, valid_loader, len(train_indices), len(val_indices)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Namespace(collections.MutableMapping):
    """Utility class to convert a (nested) dictionary into a (nested) namespace.
    >>> x = Namespace({'foo': 1, 'bar': 2})
    >>> x.foo
    1
    >>> x.bar
    2
    >>> x.baz
    Traceback (most recent call last):
        ...
    KeyError: 'baz'
    >>> x
    {'foo': 1, 'bar': 2}
    >>> (lambda **kwargs: print(kwargs))(**x)
    {'foo': 1, 'bar': 2}
    >>> x = Namespace({'foo': {'a': 1, 'b': 2}, 'bar': 3})
    >>> x.foo.a
    1
    >>> x.foo.b
    2
    >>> x.bar
    3
    >>> (lambda **kwargs: print(kwargs))(**x)
    {'foo': {'a': 1, 'b': 2}, 'bar': 3}
    >>> (lambda **kwargs: print(kwargs))(**x.foo)
    {'a': 1, 'b': 2}
    """

    def __init__(self, data):
        self._data = data

    def __getitem__(self, k):
        return self._data[k]

    def __setitem__(self, k, v):
        self._data[k] = v

    def __delitem__(self, k):
        del self._data[k]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getattr__(self, k):
        if not k.startswith('_'):
            if k not in self._data:
                return Namespace({})
            v = self._data[k]
            if isinstance(v, dict):
                v = Namespace(v)
            return v

        if k not in self.__dict__:
            raise AttributeError("'Namespace' object has no attribute '{}'".format(k))

        return self.__dict__[k]

    def __repr__(self):
        return repr(self._data)

def avg(l):
    return sum(l) / len(l) if len(l) else 0.0

def append_loss(losses, func, batch_idx, samples):
    losses.append(func(samples))

def parse_overrides(config, overrides):
    """
    Overrides the values specified in the config with values.
    config: (Nested) dictionary of parameters
    overrides: Parameters to override and new values to assign. Nested
        parameters are specified via dot notation.
    >>> parse_overrides({}, [])
    {}
    >>> parse_overrides({}, ['a'])
    Traceback (most recent call last):
      ...
    ValueError: invalid override list
    >>> parse_overrides({'a': 1}, [])
    {'a': 1}
    >>> parse_overrides({'a': 1}, ['a', 2])
    {'a': 2}
    >>> parse_overrides({'a': 1}, ['b', 2])
    Traceback (most recent call last):
      ...
    KeyError: 'b'
    >>> parse_overrides({'a': 0.5}, ['a', 'test'])
    Traceback (most recent call last):
      ...
    ValueError: could not convert string to float: 'test'
    >>> parse_overrides(
    ...    {'a': {'b': 1, 'c': 1.2}, 'd': 3},
    ...    ['d', 1, 'a.b', 3, 'a.c', 5])
    {'a': {'b': 3, 'c': 5.0}, 'd': 1}
    """
    if len(overrides) % 2 != 0:
        # print('Overrides must be of the form [PARAM VALUE]*:', ' '.join(overrides))
        raise ValueError('invalid override list')

    for param, value in zip(overrides[::2], overrides[1::2]):
        keys = param.split('.')
        params = config
        for k in keys[:-1]:
            if k not in params:
                raise KeyError(param)
            params = params[k]
        if keys[-1] not in params:
            raise KeyError(param)

        current_type = type(params[keys[-1]])
        value = current_type(value)  # cast to existing type
        params[keys[-1]] = value

    return config

def str2nparray(_string):
  return np.fromstring(_string[1:-1], dtype=float, sep=' ')

if __name__ == "__main__":
    x = Namespace({'foo': 1, 'bar': 2})
    print(x)
    x.foo = 2
    x['foo'] = 2
    print(x)
