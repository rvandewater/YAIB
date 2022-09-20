import logging
import os
import gin
import numpy as np
import torch


def save_model(model, optimizer, epoch, save_file):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def load_model_state(filepath, model, optimizer=None):
    state = torch.load(filepath)
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    logging.info('Loaded model and optimizer')


def save_config_file(log_dir):
    with open(os.path.join(log_dir, 'train_config.gin'), 'w') as f:
        f.write(gin.operative_config_str())


def get_bindings_and_params(args):
    gin_bindings = []
    log_dir = args.logdir
    if args.num_class:
        num_class = args.num_class
        gin_bindings += ['NUM_CLASSES = ' + str(num_class)]

    if args.res:
        res = args.res
        gin_bindings += ['RES = ' + str(res)]
        log_dir = os.path.join(log_dir, 'data-res_' + str(res))

    if args.res_lab:
        res_lab = args.res_lab
        gin_bindings += ['RES_LAB = ' + str(res_lab)]
        log_dir = os.path.join(log_dir, 'pre-res_' + str(res_lab))

    if args.horizon:
        if args.rs:
            horizon = args.horizon[np.random.randint(len(args.horizon))]
        else:
            horizon = args.horizon[0]
        gin_bindings += ['HORIZON  = ' + str(horizon)]
        log_dir = log_dir + '_horizon_' + str(horizon)

    if args.regularization:
        if args.rs:
            regularization = args.regularization[np.random.randint(len(args.regularization))]
        else:
            regularization = args.regularization[0]
        gin_bindings += ['L1  = ' + str(regularization)]
        log_dir = log_dir + '_l1-reg_' + str(regularization)

    if args.batch_size:
        if args.rs:
            batch_size = args.batch_size[np.random.randint(len(args.batch_size))]
        else:
            batch_size = args.batch_size[0]
        gin_bindings += ['BS = ' + str(batch_size)]
        log_dir = log_dir + '_bs_' + str(batch_size)
    if args.lr:
        if args.rs:
            lr = args.lr[np.random.randint(len(args.lr))]
        else:
            lr = args.lr[0]
        gin_bindings += ['LR = ' + str(lr)]
        log_dir = log_dir + '_lr_' + str(lr)
    if args.maxlen:
        maxlen = args.maxlen
        gin_bindings += ['MAXLEN = ' + str(maxlen)]
        log_dir = log_dir + '_maxlen_' + str(maxlen)

    if args.emb:
        if args.rs:
            emb = args.emb[np.random.randint(len(args.emb))]
        else:
            emb = args.emb[0]
        gin_bindings += ['EMB  = ' + str(emb)]
        log_dir = log_dir + '_emb_' + str(emb)

    if args.do:
        if args.rs:
            do = args.do[np.random.randint(len(args.do))]
        else:
            do = args.do[0]
        gin_bindings += ['DO  = ' + str(do)]
        log_dir = log_dir + '_do_' + str(do)

    if args.do_att:
        if args.rs:
            do_att = args.do_att[np.random.randint(len(args.do_att))]
        else:
            do_att = args.do_att[0]
        gin_bindings += ['DO_ATT  = ' + str(do_att)]
        log_dir = log_dir + '_do-att_' + str(do_att)

    if args.kernel:
        if args.rs:
            kernel = args.kernel[np.random.randint(len(args.kernel))]
        else:
            kernel = args.kernel[0]
        gin_bindings += ['KERNEL  = ' + str(kernel)]
        log_dir = log_dir + '_kernel_' + str(kernel)

    if args.depth:
        if args.rs:
            depth = args.depth[np.random.randint(len(args.depth))]
        else:
            depth = args.depth[0]

        num_leaves = 2 ** depth
        gin_bindings += ['DEPTH  = ' + str(depth)]
        gin_bindings += ['NUM_LEAVES  = ' + str(num_leaves)]
        log_dir = log_dir + '_depth_' + str(depth)

    if args.heads:
        if args.rs:
            heads = args.heads[np.random.randint(len(args.heads))]
        else:
            heads = args.heads[0]
        gin_bindings += ['HEADS  = ' + str(heads)]
        log_dir = log_dir + '_heads_' + str(heads)

    if args.latent:
        if args.rs:
            latent = args.latent[np.random.randint(len(args.latent))]
        else:
            latent = args.latent[0]
        gin_bindings += ['LATENT  = ' + str(latent)]
        log_dir = log_dir + '_latent_' + str(latent)

    if args.hidden:
        if args.rs:
            hidden = args.hidden[np.random.randint(len(args.hidden))]
        else:
            hidden = args.hidden[0]
        gin_bindings += ['HIDDEN = ' + str(hidden)]
        log_dir = log_dir + '_hidden_' + str(hidden)
    if args.subsample_data:
        if args.rs:
            subsample_data = args.subsample_data[np.random.randint(len(args.subsample_data))]
        else:
            subsample_data = args.subsample_data[0]
        gin_bindings += ['SUBSAMPLE_DATA = ' + str(subsample_data)]
        log_dir = log_dir + '_subsample-data_' + str(subsample_data)
    if args.subsample_feat:
        if args.rs:
            subsample_feat = args.subsample_feat[np.random.randint(len(args.subsample_feat))]
        else:
            subsample_feat = args.subsample_data[0]
        gin_bindings += ['SUBSAMPLE_FEAT = ' + str(subsample_feat)]
        log_dir = log_dir + '_subsample-feat_' + str(subsample_feat)

    if args.c_parameter:
        if args.rs:
            c_parameter = args.c_parameter[np.random.randint(len(args.c_parameter))]
        else:
            c_parameter = args.c_parameter[0]
        gin_bindings += ['C_PARAMETER = ' + str(c_parameter)]
        log_dir = log_dir + '_c-parameter_' + str(c_parameter)

    if args.penalty:
        if args.rs:
            penalty = args.penalty[np.random.randint(len(args.penalty))]
        else:
            penalty = args.penalty[0]
        gin_bindings += ['PENALTY = ' + "'" + str(penalty) + "'"]
        log_dir = log_dir + '_penalty_' + str(penalty)

    if args.loss_weight:
        if args.rs:
            loss_weight = args.loss_weight[np.random.randint(len(args.loss_weight))]
        else:
            loss_weight = args.loss_weight[0]
        if loss_weight == "None":
            gin_bindings += ['LOSS_WEIGHT = ' + str(loss_weight)]
            log_dir = log_dir + '_loss-weight_no_weight'
        else:
            gin_bindings += ['LOSS_WEIGHT = ' + "'" + str(loss_weight) + "'"]

    return gin_bindings, log_dir
