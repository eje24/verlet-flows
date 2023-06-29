import yaml
import resource
import math
import os

import wandb
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from utils.parsing import parse_train_args
from datasets.frame_dataset import FrameDataset, VerletFrame
from model.frame_docking.frame_docking_flow import FrameDockingVerletFlow 

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))


class AverageMeter():
    def __init__(self, types, unpooled_metrics=False, intervals=1):
        self.types = types
        self.intervals = intervals
        self.count = 0 if intervals == 1 else torch.zeros(
            len(types), intervals)
        self.acc = {t: torch.zeros(intervals) for t in types}
        self.unpooled_metrics = unpooled_metrics

    def add(self, vals, interval_idx=None):
        if self.intervals == 1:
            self.count += 1 if vals[0].dim() == 0 else len(vals[0])
            for type_idx, v in enumerate(vals):
                self.acc[self.types[type_idx]
                         ] += v.sum() if self.unpooled_metrics else v
        else:
            for type_idx, v in enumerate(vals):
                self.count[type_idx].index_add_(
                    0, interval_idx[type_idx], torch.ones(len(v)))
                if not torch.allclose(v, torch.tensor(0.0)):
                    self.acc[self.types[type_idx]].index_add_(
                        0, interval_idx[type_idx], v)

    def summary(self):
        if self.intervals == 1:
            out = {k: v.item() / self.count for k, v in self.acc.items()}
            return out
        else:
            out = {}
            for i in range(self.intervals):
                for type_idx, k in enumerate(self.types):
                    out['int' + str(i) + '_' + k] = (
                        list(self.acc.values())[type_idx][i] / self.count[type_idx][i]).item()
            return out


def train_epoch(model, loader, optimizer, device):
    model.train()
    meter = AverageMeter(['loss'])

    for (receptor, ligand, v_rot, v_tr) in tqdm(loader, total=len(loader)):
        optimizer.zero_grad()
        try:
            data = VerletFrame(receptor, ligand, v_rot, v_tr)
            log_pxv = model(data)
            loss = -torch.mean(log_pxv)
            loss.backward()
            optimizer.step()
            meter.add([loss.cpu().detach()])
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e
    return meter.summary()


def test_epoch(model, loader):
    model.eval()
    meter = AverageMeter(['loss'],
                         unpooled_metrics=True)
    for data in tqdm(loader, total=len(loader)):
        try:
            with torch.no_grad():
                log_pxv = model(VerletFrame(*data))
            loss = -torch.mean(log_pxv)
            meter.add([loss.cpu().detach()])

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    out = meter.summary()
    return out

def save_yaml_file(path, content):
    assert isinstance(
        path, str), f'path must be a string, got {path} which is a {type(path)}'
    content = yaml.dump(data=content)
    if '/' in path and os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(content)


def get_optimizer_and_scheduler(args, model, scheduler_mode='min'):
    optimizer = torch.optim.Adam(filter(
        lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.w_decay)

    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=0.7,
                                                               patience=args.scheduler_patience, min_lr=args.lr / 100)
    else:
        print('No scheduler')
        scheduler = None

    return optimizer, scheduler


def get_model(args, device):
    model_class = FrameDockingVerletFlow
    model = model_class(num_coupling_layers=args.num_coupling_layers,
                        num_conv_layers=args.num_conv_layers,
                        distance_embed_dim=args.distance_embed_dim,
                        dropout=args.dropout,
                        )
    model.to(device)
    return model


def train(args, model, optimizer, scheduler, train_loader, val_loader, run_dir):
    best_val_loss = math.inf
    best_val_inference_value = math.inf 
    best_epoch = 0
    best_val_inference_epoch = 0

    print("Starting training...")
    for epoch in range(args.n_epochs):
        if epoch % 5 == 0:
            print("Run name: ", args.run_name)
        logs = {}
        train_losses = train_epoch(
            model, train_loader, optimizer, device)
        print("Epoch {}: Training loss {:.4f}"
              .format(epoch, train_losses['loss']))

        val_losses = test_epoch(model, val_loader)
        print("Epoch {}: Validation loss {:.4f}"
              .format(epoch, val_losses['loss']))

        if args.wandb:
            logs.update({'train_' + k: v for k, v in train_losses.items()})
            logs.update({'val_' + k: v for k, v in val_losses.items()})
            logs['current_lr'] = optimizer.param_groups[0]['lr']
            wandb.log(logs, step=epoch + 1)

        state_dict = model.module.state_dict() if device.type == 'cuda' else model.state_dict()
        if val_losses['loss'] <= best_val_loss:
            best_val_loss = val_losses['loss']
            best_epoch = epoch
            torch.save(state_dict, os.path.join(run_dir, 'best_model.pt'))

        if scheduler:
            if args.val_inference_freq is not None:
                scheduler.step(best_val_inference_value)
            else:
                scheduler.step(val_losses['loss'])

        torch.save({
            'epoch': epoch,
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
        }, os.path.join(run_dir, 'last_model.pt'))

    print("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))
    print("Best inference metric {} on Epoch {}".format(
        best_val_inference_value, best_val_inference_epoch))


def main_function():
    args = parse_train_args()
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
        args.config = args.config.name

    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # construct loader
    train_loader, val_loader = FrameDataset.construct_loaders(args)

    model = get_model(args, device)
    optimizer, scheduler = get_optimizer_and_scheduler(
        args, model, scheduler_mode = 'min')
    
    if args.restart_dir:
        try:
            dict = torch.load(f'{args.restart_dir}/last_model.pt',
                              map_location=torch.device('cpu'))
            if args.restart_lr is not None:
                dict['optimizer']['param_groups'][0]['lr'] = args.restart_lr
            optimizer.load_state_dict(dict['optimizer'])
            model.module.load_state_dict(dict['model'], strict=True)
            print("Restarting from epoch", dict['epoch'])
        except Exception as e:
            print("Exception", e)
            dict = torch.load(f'{args.restart_dir}/best_model.pt',
                              map_location=torch.device('cpu'))
            model.module.load_state_dict(dict, strict=True)
            print("Due to exception had to take the best epoch and no optimiser")

    numel = sum([p.numel() for p in model.parameters()])
    print('Model with', numel, 'parameters')

    if args.wandb:
        wandb.init(
            entity='entity',
            settings=wandb.Settings(start_method="fork"),
            project=args.project,
            name=args.run_name,
            config=args
        )
        wandb.log({'numel': numel})

    # record parameters
    run_dir = os.path.join(args.log_dir, args.run_name)
    yaml_file_name = os.path.join(run_dir, 'model_parameters.yml')
    save_yaml_file(yaml_file_name, args.__dict__)
    args.device = device

    train(args, model, optimizer, scheduler,
          train_loader, val_loader, run_dir)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    main_function()
