# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: MIT
# License: licenses/ConvNeXt-V2-MIT.txt

import torch


def get_parameter_groups(model, weight_decay, skip_list=()):
    groups = {"decay": [], "no_decay": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        exempt = (
            parameter.ndim == 1 or name.endswith((".bias", ".gamma", ".beta")) or name in skip_list
        )
        groups["no_decay" if exempt else "decay"].append(parameter)
    return [
        {"params": groups["decay"], "weight_decay": weight_decay, "lr_scale": 1.0},
        {"params": groups["no_decay"], "weight_decay": 0.0, "lr_scale": 1.0},
    ]


def create_optimizer(args, model, skip_list=None):
    skip = (
        skip_list
        if skip_list is not None
        else model.no_weight_decay()
        if hasattr(model, "no_weight_decay")
        else ()
    )
    parameters = get_parameter_groups(model, args.weight_decay, skip)
    if args.opt == "sgd":
        return torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, nesterov=True)
    constructor = torch.optim.AdamW if args.opt == "adamw" else torch.optim.Adam
    return constructor(
        parameters,
        lr=args.lr,
        eps=args.opt_eps,
        betas=tuple(args.opt_betas) if args.opt_betas is not None else (0.9, 0.999),
    )
