import yaml
import torch
import torchvision.models as models
from robustbench.utils import load_model as load_clf

from score_sde.losses import get_optimizer
from score_sde.models import utils as mutils
from score_sde.models.ema import ExponentialMovingAverage
from score_sde import sde_lib
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from utils import dict2namespace, restore_checkpoint


def load_models(args, model_src, device):
    if args.dataset == 'cifar10':
        # Diffusion model
        with open('./diffusion_configs/cifar10.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
        config = dict2namespace(config)
        diffusion = mutils.create_model(config)
        optimizer = get_optimizer(config, diffusion.parameters())
        ema = ExponentialMovingAverage(
            diffusion.parameters(), decay=config.model.ema_rate)
        state = dict(step=0, optimizer=optimizer, model=diffusion, ema=ema)
        restore_checkpoint(model_src, state, device)
        ema.copy_to(diffusion.parameters())
        diffusion.eval().to(device)

        # Underlying classifier
        clf = load_clf(model_name='Standard',
                       dataset='cifar10').to(device).eval()
    elif args.dataset == 'imagenet':
        with open('./diffusion_configs/imagenet.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
        config = dict2namespace(config)
        model_config = model_and_diffusion_defaults()
        model_config.update(vars(config.model))
        diffusion, _ = create_model_and_diffusion(**model_config)
        diffusion.load_state_dict(torch.load(model_src, map_location='cpu'))
        diffusion.eval().to(device)

        # Underlying classifier
        clf = models.resnet50(pretrained=True).to(device).eval()
    return clf, diffusion
