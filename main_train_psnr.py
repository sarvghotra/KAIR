import os.path
import math
import argparse
import time
import random
import numpy as np
import lpips
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model
from torch.distributed.elastic.multiprocessing.errors import record

'''
# --------------------------------------------
# training code for MSRResNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''

# @record
def main(json_path='options/train_msrresnet_psnr.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=True)
    parser.add_argument('--resume_crashed_training', action='store_true')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist
    opt['resume_crashed_training'] = parser.parse_args().resume_crashed_training

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    opt['num_gpu'] = 0
    if dist.is_available() and dist.is_initialized():
        opt['num_gpu'] = opt['world_size']
    else:
        opt['num_gpu'] = torch.cuda.device_count()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    current_step = 0
    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    if opt['path']['pretrained_netG'] is None or opt['resume_crashed_training']:
        init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
        init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
        init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
        init_iter_schedulerG, init_path_schedulerG = option.find_last_checkpoint(opt['path']['models'], net_type='schedulerG')

        opt['path']['pretrained_netG'] = init_path_G
        opt['path']['pretrained_netE'] = init_path_E
        opt['path']['pretrained_optimizerG'] = init_path_optimizerG
        opt['path']['pretrained_schedulerG'] = init_path_schedulerG

        if "gan" in opt['model']:
            init_iter_D, init_path_D = option.find_last_checkpoint(opt['path']['models'], net_type='D')
            init_iter_optimizerD, init_path_optimizerD = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerD')
            opt['path']['pretrained_netD'] = init_path_D
            opt['path']['pretrained_optimizerD'] = init_path_optimizerD

        current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG, init_iter_schedulerG)
    elif opt['train']['continue_training']:
        current_step = int(os.path.basename(opt['path']['pretrained_netG']).split('_')[0])

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.info(option.dict2str(opt))
        tb_path = os.path.join(opt['path']['tb_log'], opt['task'])
        tb_writer = SummaryWriter(tb_path)

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)

    if opt['rank'] == 0:
        logger.info('Random seed: {}'.format(seed))
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    test_sets_info = []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif 'test' in phase:
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=8,
                                     drop_last=False, pin_memory=True)
            set_info = {}
            set_info['name'] = dataset_opt["name"]
            set_info['phase'] = phase
            set_info['loader'] = test_loader
            test_sets_info.append(set_info)
        elif "save" in phase:
            save_test_set = define_Dataset(dataset_opt)
            save_test_loader = DataLoader(save_test_set, batch_size=1,
                                     shuffle=False, num_workers=8,
                                     drop_last=False, pin_memory=True)
            set_info = {}
            set_info['name'] = dataset_opt["name"]
            set_info['phase'] = phase
            set_info['loader'] = save_test_loader
            test_sets_info.append(set_info)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()
    if opt['rank'] == 0:
        # logger.info(model.info_network())
        # logger.info(model.info_params())
        logger.info('G Params number: {}'.format(sum(map(lambda x: x.numel(), model.netG.parameters()))) + '\n')
        if hasattr(model, 'netD'):
            logger.info('D Params number: {}'.format(sum(map(lambda x: x.numel(), model.netD.parameters()))) + '\n')
        logger.info('Number of GPUs is: ' + str(opt['num_gpu']))
        logger.info("Device count per node: " + str(torch.cuda.device_count()))
        loss_fn = lpips.LPIPS(net='alex')
        loss_fn.cuda()

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    for epoch in range(1000000):  # keep running
        if opt['dist']:
            train_sampler.set_epoch(epoch)

        for i, train_data in enumerate(train_loader):

            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}> '.format(epoch, current_step)
                for k, v in logs.items():  # merge log information into message
                    avg_v = sum(v)/len(v) if len(v) > 0 else v
                    message += '{:s}: {:.3e} '.format(k, avg_v)
                    tb_writer.add_scalar('Loss/train/' + k, avg_v, current_step)    # Tensorboard logging

                lr_logs = model.current_learning_rate()
                for k, v in lr_logs.items():
                    tb_writer.add_scalar('LR/' + k, v, current_step)    # Tensorboard logging
                    message += '{:s}: {:.3e} '.format(k, v)

                tb_writer.flush()
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:
                all_testsets_avg_psnr = 0.0
                all_testsets_avg_lpips = 0.0
                nb_testsets = 0
                for t in test_sets_info:
                    test_loader = t['loader']
                    testset_name = t['name']
                    phase = t['phase']
                    avg_psnr = 0.0
                    avg_lpips = 0.0
                    idx = 0
                    if "save" not in phase:
                        nb_testsets += 1

                    for test_data in test_loader:
                        image_name_ext = os.path.basename(test_data['L_path'][0])
                        img_name, ext = os.path.splitext(image_name_ext)

                        model.feed_data(test_data)
                        model.test()

                        # -----------------------
                        # calculate LPIPS
                        # -----------------------
                        if "save" not in phase:
                            curr_lpips_diff = loss_fn.forward(model.E, model.H)
                            avg_lpips += curr_lpips_diff.item()

                        visuals = model.current_visuals()
                        E_img = util.tensor2uint(visuals['E'])
                        H_img = util.tensor2uint(visuals['H'])

                        # -----------------------
                        # save estimated image E
                        # -----------------------
                        if "save" in phase:
                            img_dir = os.path.join(opt['path']['images'], img_name)
                            util.mkdir(img_dir)
                            save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                            util.imsave(E_img, save_img_path)
                            continue

                        # -----------------------
                        # calculate PSNR
                        # -----------------------
                        current_psnr = util.calculate_psnr(E_img, H_img, border=border)
                        # Safety measure for inf psnr
                        if current_psnr > 300:
                            continue

                        idx += 1
                        #logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))
                        avg_psnr += current_psnr

                    if idx > 0:
                        avg_psnr = avg_psnr / idx
                        avg_lpips = avg_lpips / idx
                    if "save" not in phase:
                        # testing log
                        logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB LPIPS: {:<.4f} {}\n'.format(epoch, current_step, avg_psnr, avg_lpips, testset_name))
                        tb_writer.add_scalar('PSNR/val/' + testset_name, avg_psnr, current_step)
                        tb_writer.add_scalar('LPIPS/val/' + testset_name, avg_lpips, current_step)

                    all_testsets_avg_psnr += avg_psnr
                    all_testsets_avg_lpips += avg_lpips

                if nb_testsets > 0:
                    all_testsets_avg_psnr = all_testsets_avg_psnr / nb_testsets
                    all_testsets_avg_lpips = all_testsets_avg_lpips / nb_testsets
                # testing log
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB LPIPS: {:<.4f}\n'.format(epoch, current_step, all_testsets_avg_psnr, all_testsets_avg_lpips))
                tb_writer.add_scalar('PSNR/val/all_tests_avg', all_testsets_avg_psnr, current_step)
                tb_writer.add_scalar('LPIPS/val/all_tests_avg', all_testsets_avg_lpips, current_step)


if __name__ == '__main__':
    main()
