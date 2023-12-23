import yaml
import resource
import math
import os
from tqdm import tqdm

import wandb
import torch

# torch.multiprocessing.set_sharing_strategy("file_system")

from utils.parsing import display_args, parse_args
from model.flow import FlowWrapper, VerletFlow
from datasets.dist import GMM, Gaussian, VerletGaussian, VerletGMM

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))


class AverageMeter:
    def __init__(self, types, unpooled_metrics=False, intervals=1):
        self.types = types
        self.intervals = intervals
        self.count = 0 if intervals == 1 else torch.zeros(len(types), intervals)
        self.acc = {t: torch.zeros(intervals) for t in types}
        self.unpooled_metrics = unpooled_metrics

    def add(self, vals, interval_idx=None):
        if self.intervals == 1:
            self.count += 1 if vals[0].dim() == 0 else len(vals[0])
            for type_idx, v in enumerate(vals):
                self.acc[self.types[type_idx]] += (
                    v.sum() if self.unpooled_metrics else v
                )
        else:
            for type_idx, v in enumerate(vals):
                self.count[type_idx].index_add_(
                    0, interval_idx[type_idx], torch.ones(len(v))
                )
                if not torch.allclose(v, torch.tensor(0.0)):
                    self.acc[self.types[type_idx]].index_add_(
                        0, interval_idx[type_idx], v
                    )

    def summary(self):
        if self.intervals == 1:
            out = {k: v.item() / self.count for k, v in self.acc.items()}
            return out
        else:
            out = {}
            for i in range(self.intervals):
                for type_idx, k in enumerate(self.types):
                    out["int" + str(i) + "_" + k] = (
                        list(self.acc.values())[type_idx][i] / self.count[type_idx][i]
                    ).item()
            return out


def train_epoch(flow_wrapper, optimizer, device, num_train, batch_size, num_integrator_steps):
    flow_wrapper.train()
    meter = AverageMeter(["loss"])

    num_batches = math.ceil(num_train / batch_size)
    for _ in tqdm(range(num_batches), total=num_batches):
        optimizer.zero_grad()
        try:
            logp = flow_wrapper(batch_size, num_integrator_steps)
            loss = -torch.mean(logp)
            loss.backward()
            optimizer.step()
            meter.add([loss.cpu().detach()])
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory, skipping batch")
                for p in flow_wrapper.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif "Input mismatch" in str(e):
                print("| WARNING: weird torch_cluster error, skipping batch")
                for p in flow_wrapper.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e
    return meter.summary()


def test_epoch(flow_wrapper, num_test, batch_size, num_integrator_steps):
    flow_wrapper.eval()
    meter = AverageMeter(["loss"], unpooled_metrics=True)

    num_batches = math.ceil(num_test / batch_size)
    for _ in tqdm(range(num_batches), total=num_batches):
        try:
            with torch.no_grad():
                logp = flow_wrapper(batch_size, num_integrator_steps)
            loss = -torch.mean(logp)
            meter.add([loss.cpu().detach()])

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory, skipping batch")
                for p in flow_wrapper.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif "Input mismatch" in str(e):
                print("| WARNING: weird torch_cluster error, skipping batch")
                for p in flow_wrapper.parameters():
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
        path, str
    ), f"path must be a string, got {path} which is a {type(path)}"
    content = yaml.dump(data=content)
    if (
        "/" in path
        and os.path.dirname(path)
        and not os.path.exists(os.path.dirname(path))
    ):
        os.makedirs(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(content)


def get_optimizer_and_scheduler(args, model, scheduler_mode="min"):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.w_decay,
    )

    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_mode,
            factor=0.7,
            patience=args.scheduler_patience,
            min_lr=args.lr / 100,
        )
    else:
        print("No scheduler")
        scheduler = None

    return optimizer, scheduler


def get_model(args, device):
    # Initialize model
    verlet_flow = VerletFlow(2, 5, 10)

    # Initialize sampleable source distribution
    q_sampleable = Gaussian(torch.zeros(2, device=device), torch.eye(2, device=device))
    p_sampleable = Gaussian(torch.zeros(2, device=device), torch.eye(2, device=device))
    source = VerletGaussian(
        q_sampleable = q_sampleable,
        p_sampleable = p_sampleable,
    )

    # Initialize target density
    q_density = GMM(device=device, nmode=3, xlim=1.0, scale=1.0)
    p_density = Gaussian(torch.zeros(2, device=device), torch.eye(2, device=device))
    target = VerletGMM(
        q_density = q_density,
        p_density = p_density,
    )

    # Initialize flow wrapper
    model = FlowWrapper(
        flow = verlet_flow,
        source = source,
        target = target,
    )
    model.to(device)

    return model


def train(args, flow_wrapper, optimizer, scheduler, run_dir):
    best_val_loss = math.inf
    best_val_inference_value = math.inf
    best_epoch = 0
    best_val_inference_epoch = 0

    print("Starting training...")
    for epoch in range(args.n_epochs):
        if epoch % 5 == 0:
            print("Run name: ", args.run_name)
        logs = {}
        train_losses = train_epoch(flow_wrapper, optimizer, device, args.num_train, args.batch_size, args.num_integrator_steps)
        print("Epoch {}: Training loss {:.4f}".format(epoch, train_losses["loss"]))

        val_losses = test_epoch(flow_wrapper, args.num_val, args.batch_size, args.num_integrator_steps)
        print("Epoch {}: Validation loss {:.4f}".format(epoch, val_losses["loss"]))

        if args.wandb:
            logs.update({"train_" + k: v for k, v in train_losses.items()})
            logs.update({"val_" + k: v for k, v in val_losses.items()})
            logs["current_lr"] = optimizer.param_groups[0]["lr"]
            wandb.log(logs, step=epoch + 1)

        state_dict = flow_wrapper.state_dict() if device.type == "cuda" else flow_wrapper.state_dict()
        if val_losses["loss"] <= best_val_loss:
            best_val_loss = val_losses["loss"]
            best_epoch = epoch
            torch.save(state_dict, os.path.join(run_dir, "best_model.pt"))

        if scheduler:
            if args.val_inference_freq is not None:
                scheduler.step(best_val_inference_value)
            else:
                scheduler.step(val_losses["loss"])

        torch.save(
            {
                "epoch": epoch,
                "model": state_dict,
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(run_dir, "last_model.pt"),
        )

    print("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))
    print(
        "Best inference metric {} on Epoch {}".format(
            best_val_inference_value, best_val_inference_epoch
        )
    )


def main_function():
    args = parse_args()
    display_args(args)
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

    model = get_model(args, device)
    optimizer, scheduler = get_optimizer_and_scheduler(
        args, model, scheduler_mode="min"
    )

    if args.restart_dir:
        try:
            dict = torch.load(
                f"{args.restart_dir}/last_model.pt", map_location=torch.device("cpu")
            )
            if args.restart_lr is not None:
                dict["optimizer"]["param_groups"][0]["lr"] = args.restart_lr
            optimizer.load_state_dict(dict["optimizer"])
            model.load_state_dict(dict["model"], strict=True)
            print("Restarting from epoch", dict["epoch"])
        except Exception as e:
            print("Exception", e)
            dict = torch.load(
                f"{args.restart_dir}/best_model.pt", map_location=torch.device("cpu")
            )
            model.load_state_dict(dict, strict=True)
            print("Due to exception had to take the best epoch and no optimiser")

    numel = sum([p.numel() for p in model.parameters()])
    print("Model with", numel, "parameters")

    if args.wandb:
        wandb.init(
            entity="entity",
            settings=wandb.Settings(start_method="fork"),
            project=args.project,
            name=args.run_name,
            config=args,
        )
        wandb.log({"numel": numel})

    # record parameters
    run_dir = os.path.join(args.log_dir, args.run_name)
    yaml_file_name = os.path.join(run_dir, "model_parameters.yml")
    save_yaml_file(yaml_file_name, args.__dict__)
    args.device = device

    train(args, model, optimizer, scheduler, run_dir)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main_function()
