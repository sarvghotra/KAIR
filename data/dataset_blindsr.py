import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
import os
from utils import utils_blindsr as blindsr


class DatasetSLBlindSR(data.Dataset):
    '''
    # -----------------------------------------
    # dataset for BSRGAN
    # -----------------------------------------
    '''
    def __init__(self, opt):
        super(DatasetSLBlindSR, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.shuffle_prob = opt['shuffle_prob'] if opt['shuffle_prob'] else 0.1
        self.use_sharp = opt['use_sharp'] if opt['use_sharp'] else False
        self.degradation_type = opt['degradation_type'] if opt['degradation_type'] else 'bsrgan'
        self.lq_patchsize = self.opt['lq_patchsize'] if self.opt['lq_patchsize'] else 64
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else self.lq_patchsize*self.sf

        train_img_files_hr = util.read_dirs(opt['dataroot_H'], opt['cache_dir'])
        non_empty_dataroot_L = [x for x in opt['dataroot_L'] if x is not None]
        train_img_files_L = util.read_dirs(non_empty_dataroot_L, opt['cache_dir'])
        # merging all lists of filepaths into one list
        self.paths_H = []
        self.paths_L = []
        i = 0
        for data_name, d, data_L, d_size in zip(opt['dataroot_H'], train_img_files_hr, opt['dataroot_L'], opt['data_sizes']):
            # TODO
            # FIXME: Move it to config
            # if "FFHQ" in data_name:
            #     d = d[:100]
            # elif "SCUT-CTW1500" in data_name:
            #     d = d[:100]
            # elif "Crawl" in data_name:
            #     random.shuffle(d)
            #     d = d[:4000]
            max_imgs = d_size
            if d_size is None:
                max_imgs = len(d)

            if data_L is not None:
                assert len(d) == len(train_img_files_L[i]), "LR and HR data size must be same"
                d = d[:max_imgs]
                self.paths_L.extend(train_img_files_L[i][:max_imgs])
                i += 1
            else:
                d = d[:max_imgs]
                self.paths_L.extend([None] * len(d))

            self.paths_H.extend(d)

        print(len(self.paths_H))

#        for n, v in enumerate(self.paths_H):
#            if 'face' in v:
#                del self.paths_H[n]
#        time.sleep(1)
        assert self.paths_H, 'Error: H path is empty.'
        assert len(self.paths_H) == len(self.paths_L), 'Error: LR and HR have different sizes'

    def __getitem__(self, index):
        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        img_name, ext = os.path.splitext(os.path.basename(H_path))
        H, W, C = img_H.shape

        if H < self.patch_size or W < self.patch_size:
            img_H = np.tile(np.random.randint(0, 256, size=[1, 1, self.n_channels], dtype=np.uint8), (self.patch_size, self.patch_size, 1))

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, C = img_H.shape

            rnd_h_H = random.randint(0, max(0, H - self.patch_size))
            rnd_w_H = random.randint(0, max(0, W - self.patch_size))
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            L_path = self.paths_L[index]
            if L_path is not None:
                # ------------------------------------
                # get L image
                # ------------------------------------
                img_L = util.imread_uint(L_path, self.n_channels)
                img_L = img_L[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]


            if 'face' in img_name:
                mode = random.choice([0, 4])
                img_H = util.augment_img(img_H, mode=mode)
                if L_path is not None:
                    img_L = util.augment_img(img_L, mode=mode)
            else:
                mode = random.randint(0, 7)
                img_H = util.augment_img(img_H, mode=mode)
                if L_path is not None:
                    img_L = util.augment_img(img_L, mode=mode)

            img_H = util.uint2single(img_H)
            if L_path is not None:
                img_L = util.uint2single(img_L)
            else:
                img_L = None

            if self.degradation_type == 'bsrgan':
                img_L, img_H = blindsr.degradation_bsrgan(img_H, self.sf, lq_img=img_L, lq_patchsize=self.lq_patchsize, isp_model=None)
            elif self.degradation_type == 'bsrgan_plus':
                img_L, img_H = blindsr.degradation_bsrgan_plus(img_H, self.sf, lq_img=img_L, shuffle_prob=self.shuffle_prob, use_sharp=self.use_sharp, lq_patchsize=self.lq_patchsize)

        else:
            img_H = util.uint2single(img_H)
            if self.degradation_type == 'bsrgan':
                img_L, img_H = blindsr.degradation_bsrgan(img_H, self.sf, lq_patchsize=self.lq_patchsize, isp_model=None)
            elif self.degradation_type == 'bsrgan_plus':
                img_L, img_H = blindsr.degradation_bsrgan_plus(img_H, self.sf, shuffle_prob=self.shuffle_prob, use_sharp=self.use_sharp, lq_patchsize=self.lq_patchsize)

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

        # img_H_out = util.tensor2uint(img_H)
        # img_L_out = util.tensor2uint(img_L)
        # util.imsave(img_H_out, "/tmp/imgs/" + img_name + "_h.png")
        # #img_name, ext = os.path.splitext(os.path.basename(L_path))
        # util.imsave(img_L_out, "/tmp/imgs/" + img_name + "_l.png")
        # print("saved")

        if L_path is None:
            L_path = H_path

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)


class DatasetBlindSR(data.Dataset):
    '''
    # -----------------------------------------
    # dataset for BSRGAN
    # -----------------------------------------
    '''
    def __init__(self, opt):
        super(DatasetBlindSR, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.shuffle_prob = opt['shuffle_prob'] if opt['shuffle_prob'] else 0.1
        self.use_sharp = opt['use_sharp'] if opt['use_sharp'] else False
        self.degradation_type = opt['degradation_type'] if opt['degradation_type'] else 'bsrgan'
        self.lq_patchsize = self.opt['lq_patchsize'] if self.opt['lq_patchsize'] else 64
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else self.lq_patchsize*self.sf

        train_img_files = util.read_dirs(opt['dataroot_H'], opt['cache_dir'])
        # merging all lists of filepaths into one list
        self.paths_H = []
        for data_name, d in zip(opt['dataroot_H'], train_img_files):
            # TODO
            # FIXME: Move it to config
            if "ffhq" in data_name.lower():
                d = d[:2000]
            elif "SCUT-CTW1500" in data_name:
                d = d[:100]
            elif "crawl" in data_name.lower():
                random.shuffle(d)
                d = d[:40000]
            self.paths_H.extend(d)

        print(len(self.paths_H))

#        for n, v in enumerate(self.paths_H):
#            if 'face' in v:
#                del self.paths_H[n]
#        time.sleep(1)
        assert self.paths_H, 'Error: H path is empty.'

    def __getitem__(self, index):

        L_path = None

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        img_name, ext = os.path.splitext(os.path.basename(H_path))
        H, W, C = img_H.shape

        if H < self.patch_size or W < self.patch_size:
            img_H = np.tile(np.random.randint(0, 256, size=[1, 1, self.n_channels], dtype=np.uint8), (self.patch_size, self.patch_size, 1))

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, C = img_H.shape

            rnd_h_H = random.randint(0, max(0, H - self.patch_size))
            rnd_w_H = random.randint(0, max(0, W - self.patch_size))
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            if 'face' in img_name:
                mode = random.choice([0, 4])
                img_H = util.augment_img(img_H, mode=mode)
            else:
                mode = random.randint(0, 7)
                img_H = util.augment_img(img_H, mode=mode)

            img_H = util.uint2single(img_H)
            if self.degradation_type == 'bsrgan':
                img_L, img_H = blindsr.degradation_bsrgan(img_H, self.sf, lq_patchsize=self.lq_patchsize, isp_model=None)
            elif self.degradation_type == 'bsrgan_plus':
                img_L, img_H = blindsr.degradation_bsrgan_plus(img_H, self.sf, shuffle_prob=self.shuffle_prob, use_sharp=self.use_sharp, lq_patchsize=self.lq_patchsize)

        else:
            img_H = util.uint2single(img_H)
            if self.degradation_type == 'bsrgan':
                img_L, img_H = blindsr.degradation_bsrgan(img_H, self.sf, lq_patchsize=self.lq_patchsize, isp_model=None)
            elif self.degradation_type == 'bsrgan_plus':
                img_L, img_H = blindsr.degradation_bsrgan_plus(img_H, self.sf, shuffle_prob=self.shuffle_prob, use_sharp=self.use_sharp, lq_patchsize=self.lq_patchsize)

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

        if L_path is None:
            L_path = H_path

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)


class DatasetBlindSRLRHR(data.Dataset):
    '''
    # -----------------------------------------
    # dataset for BSRGAN
    # -----------------------------------------
    '''
    def __init__(self, opt):
        super(DatasetBlindSRLRHR, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = int(opt['scale']) if opt['scale'] else 4
        self.shuffle_prob = opt['shuffle_prob'] if opt['shuffle_prob'] else 0.1
        self.use_sharp = opt['use_sharp'] if opt['use_sharp'] else False
        self.degradation_type = opt['degradation_type'] if opt['degradation_type'] else 'bsrgan'
        self.lq_patchsize = self.opt['lq_patchsize'] if self.opt['lq_patchsize'] else 64
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else self.lq_patchsize*self.sf

        train_img_files = util.read_dirs(opt['dataroot_H'], opt['cache_dir'])
        non_empty_dataroot_L = [x for x in opt['dataroot_L'] if x is not None]
        train_img_files_L = util.read_dirs(non_empty_dataroot_L, opt['cache_dir'])
        # merging all lists of filepaths into one list
        self.paths_H = []
        self.paths_L = []
        self.cum_data_sizes = [0]
        self.cum_actual_data_sizes = [0]
        i = 0
        for data_name, data_name_lr, data_size, max_act_data_size, d in zip(opt['dataroot_H'], opt['dataroot_L'], opt['data_sizes'], opt['max_actual_data_size'], train_img_files):
            disk_data_size = len(d)
            if "crawl" in data_name.lower():
                random.shuffle(d)

            if max_act_data_size is not None:
                d = d[:max_act_data_size]

            max_imgs = data_size
            self.cum_actual_data_sizes.append(len(d))
            if data_size is None:
                max_imgs = len(d)

            self.paths_H.extend(d)

            if data_name_lr is None:
                # dummy enteries to keep lr, hr indices same
                self.paths_L.extend([None] * len(d))
            else:
                d_lr = train_img_files_L[i]
                if max_act_data_size is not None:
                    d_lr = d_lr[:max_act_data_size]

                assert len(d_lr) == len(d), "{} {} {}".format(len(d_lr), len(d), data_name)
                self.paths_L.extend(d_lr)
                i += 1

            self.cum_data_sizes.append(max_imgs)
            print("=====================\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n=========================".format(data_name, data_name_lr, data_size, max_act_data_size, disk_data_size, self.paths_H[-1], self.paths_L[-1]))

        for k in range(1, len(self.cum_actual_data_sizes)):
            self.cum_actual_data_sizes[k] = self.cum_actual_data_sizes[k-1] + self.cum_actual_data_sizes[k]
            self.cum_data_sizes[k] = self.cum_data_sizes[k-1] + self.cum_data_sizes[k]

        print("=====================\n sampling data size: {}\n actual data size: {}\n=========================".format( self.cum_data_sizes[-1], len(self.paths_H)))
        assert self.paths_H, 'Error: H path is empty.'

    def _get_img_paths(self, index):
        for i, boundry in enumerate(self.cum_data_sizes):
            if index < boundry:
                data_i = i
                break

        data_sampling_size = self.cum_data_sizes[data_i] - self.cum_data_sizes[data_i - 1]
        data_actual_size = self.cum_actual_data_sizes[data_i] - self.cum_actual_data_sizes[data_i - 1]
        index = (index - self.cum_data_sizes[data_i - 1])


        # sample more than data size
        if data_sampling_size >= data_actual_size:
            # map down the index
            index = index % data_actual_size
        else:
            # expand to bigger datapoint options
            index = random.choice(range((index) * (data_actual_size // data_sampling_size), (index+1) * (data_actual_size // data_sampling_size)))

        index = index + self.cum_actual_data_sizes[data_i - 1]
        return self.paths_L[index], self.paths_H[index]

    def __getitem__(self, index):
        L_path, H_path = self._get_img_paths(index)

        # ------------------------------------
        # get H image
        # ------------------------------------
        img_H = util.imread_uint(H_path, self.n_channels)
        img_name, ext = os.path.splitext(os.path.basename(H_path))
        H, W, C = img_H.shape

        if H < self.patch_size or W < self.patch_size:
            img_H = np.tile(np.random.randint(0, 256, size=[1, 1, self.n_channels], dtype=np.uint8), (self.patch_size, self.patch_size, 1))

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, C = img_H.shape

            rnd_h_H = random.randint(0, max(0, H - self.patch_size))
            rnd_w_H = random.randint(0, max(0, W - self.patch_size))
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            if L_path is not None:
                # ------------------------------------
                # get L image
                # ------------------------------------
                img_L = util.imread_uint(L_path, self.n_channels)
                img_L = img_L[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            if 'face' in img_name:
                mode = random.choice([0, 4])
                img_H = util.augment_img(img_H, mode=mode)
                if L_path is not None:
                    img_L = util.augment_img(img_L, mode=mode)
            else:
                mode = random.randint(0, 7)
                img_H = util.augment_img(img_H, mode=mode)
                if L_path is not None:
                    img_L = util.augment_img(img_L, mode=mode)

            img_H = util.uint2single(img_H)
            if L_path is not None:
                img_L = util.uint2single(img_L)
            else:
                img_L = None

            if self.degradation_type == 'bsrgan':
                img_L, img_H = blindsr.degradation_bsrgan(img_H, self.sf, lq_img=img_L, lq_patchsize=self.lq_patchsize, isp_model=None)
            elif self.degradation_type == 'bsrgan_plus':
                img_L, img_H = blindsr.degradation_bsrgan_plus(img_H, self.sf, lq_img=img_L, shuffle_prob=self.shuffle_prob, use_sharp=self.use_sharp, lq_patchsize=self.lq_patchsize)

        else:
            img_H = util.uint2single(img_H)
            if self.degradation_type == 'bsrgan':
                img_L, img_H = blindsr.degradation_bsrgan(img_H, self.sf, lq_patchsize=self.lq_patchsize, isp_model=None)
            elif self.degradation_type == 'bsrgan_plus':
                img_L, img_H = blindsr.degradation_bsrgan_plus(img_H, self.sf, shuffle_prob=self.shuffle_prob, use_sharp=self.use_sharp, lq_patchsize=self.lq_patchsize)

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

        # img_H_out = util.tensor2uint(img_H)
        # img_L_out = util.tensor2uint(img_L)
        # util.imsave(img_H_out, "/tmp/imgs/" + img_name + "_h.png")
        # #img_name, ext = os.path.splitext(os.path.basename(L_path))
        # util.imsave(img_L_out, "/tmp/imgs/" + img_name + "_l.png")
        # print("saved")

        if L_path is None:
            L_path = H_path

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return self.cum_data_sizes[-1]
