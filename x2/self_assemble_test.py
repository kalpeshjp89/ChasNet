import os
import sys
import logging
import time
import argparse
import numpy as np
from collections import OrderedDict
import torch
import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model

# options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))
opt = option.dict_to_nonedict(opt)

util.setup_logger(None, opt['path']['log'], 'test.log', level=logging.INFO, screen=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))
# Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

# Create model
model = create_model(opt)

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    for data in test_loader:
        need_HR = False if test_loader.dataset.opt['dataroot_HR'] is None else True
        s=[]
        d1=data['LR']
        d2=torch.rot90(d1,1,[2,3]) #90
        d3=torch.rot90(d2,1,[2,3]) #180
        d4=torch.rot90(d3,1,[2,3]) #270
        d5=torch.flip(d1,[2])
        d6=torch.rot90(d5,1,[2,3]) #90
        d7=torch.rot90(d6,1,[2,3]) #180
        d8=torch.rot90(d7,1,[2,3]) #270
        
        img_path = data['LR_path'][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        data['LR']=d2
        model.feed_data(data, need_HR=need_HR)
        model.test()  # test
        visuals = model.get_current_visuals(need_HR=False)
        s.append(torch.rot90(visuals['SR'],-1,[1,2]))

        data['LR']=d3
        model.feed_data(data, need_HR=need_HR)
        model.test()  # test
        visuals = model.get_current_visuals(need_HR=False)
        s.append(torch.rot90(torch.rot90(visuals['SR'],-1,[1,2]),-1,[1,2]))

        data['LR']=d4
        model.feed_data(data, need_HR=need_HR)
        model.test()  # test
        visuals = model.get_current_visuals(need_HR=False)
        s.append(torch.rot90(torch.rot90(torch.rot90(visuals['SR'],-1,[1,2]),-1,[1,2]),-1,[1,2]))

        data['LR']=d5
        model.feed_data(data, need_HR=need_HR)
        model.test()  # test
        visuals = model.get_current_visuals(need_HR=False)
        s.append(torch.flip(visuals['SR'],[1]))

        data['LR']=d6
        model.feed_data(data, need_HR=need_HR)
        model.test()  # test
        visuals = model.get_current_visuals(need_HR=False)
        s.append(torch.flip(torch.rot90(visuals['SR'],-1,[1,2]),[1]))

        data['LR']=d7
        model.feed_data(data, need_HR=need_HR)
        model.test()  # test
        visuals = model.get_current_visuals(need_HR=False)
        s.append(torch.flip(torch.rot90(torch.rot90(visuals['SR'],-1,[1,2]),-1,[1,2]),[1]))

        data['LR']=d8
        model.feed_data(data, need_HR=need_HR)
        model.test()  # test
        visuals = model.get_current_visuals(need_HR=False)
        s.append(torch.flip(torch.rot90(torch.rot90(torch.rot90(visuals['SR'],-1,[1,2]),-1,[1,2]),-1,[1,2]),[1]))

        data['LR']=d1
        model.feed_data(data, need_HR=need_HR)
        model.test()  # test
        visuals = model.get_current_visuals(need_HR=need_HR)
        s.append(visuals['SR'])

        final_SR = (s[0] + s[1] + s[3] + s[4] + s[5] + s[6] + s[7])//8
        final_SR = torch.mean(torch.stack(s), dim=0)

        #util.save_img(util.tensor2img(s[0]), os.path.join(dataset_dir, img_name + '1.png'))
        #util.save_img(util.tensor2img(s[1]), os.path.join(dataset_dir, img_name + '2.png'))
        #util.save_img(util.tensor2img(s[2]), os.path.join(dataset_dir, img_name + '3.png'))
        #util.save_img(util.tensor2img(s[3]), os.path.join(dataset_dir, img_name + '4.png'))
        #util.save_img(util.tensor2img(s[4]), os.path.join(dataset_dir, img_name + '5.png'))
        #util.save_img(util.tensor2img(s[5]), os.path.join(dataset_dir, img_name + '6.png'))
        #util.save_img(util.tensor2img(s[6]), os.path.join(dataset_dir, img_name + '7.png'))
        #util.save_img(util.tensor2img(s[7]), os.path.join(dataset_dir, img_name + '8.png'))
        sr_img = util.tensor2img(final_SR)  # uint8

        # save images
        suffix = opt['suffix']
        if suffix:
            save_img_path = os.path.join(dataset_dir, img_name + suffix + '.jpg')
        else:
            save_img_path = os.path.join(dataset_dir, img_name + '.jpg')
        util.save_img(sr_img, save_img_path)

        # calculate PSNR and SSIM
        if need_HR:
            gt_img = util.tensor2img(visuals['HR'])
            gt_img = gt_img / 255.
            sr_img = sr_img / 255.

            crop_border = test_loader.dataset.opt['scale']
            cropped_sr_img = sr_img[crop_border:-crop_border, crop_border:-crop_border, :]
            cropped_gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, :]

            psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
            ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)

            if gt_img.shape[2] == 3:  # RGB image
                sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                cropped_sr_img_y = sr_img_y[crop_border:-crop_border, crop_border:-crop_border]
                cropped_gt_img_y = gt_img_y[crop_border:-crop_border, crop_border:-crop_border]
                psnr_y = util.calculate_psnr(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                ssim_y = util.calculate_ssim(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)
                logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'\
                    .format(img_name, psnr, ssim, psnr_y, ssim_y))
            else:
                logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(img_name, psnr, ssim))
        else:
            logger.info(img_name)

    if need_HR:  # metrics
        # Average PSNR/SSIM results
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        logger.info('----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n'\
                .format(test_set_name, ave_psnr, ave_ssim))
        if test_results['psnr_y'] and test_results['ssim_y']:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            logger.info('----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n'\
                .format(ave_psnr_y, ave_ssim_y))
