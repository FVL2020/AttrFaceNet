import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import InpaintingModel, AttrModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR, EdgeAccuracy
np.set_printoptions(precision=4, suppress=True)

class AttrFaceNet():
    def __init__(self, config):
        self.config = config

        if config.MODEL == 1:
            model_name = 'attr'
        elif config.MODEL == 2:
            model_name = 'inpaint'
        elif config.MODEL == 3:
            model_name = 'attr_inpaint'
        elif config.MODEL == 4:
            model_name = 'joint'

        self.debug = False
        self.model_name = model_name
        self.attr_model = AttrModel(config).to(config.DEVICE)
        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)
        selected_attr = ['Male', 'Big_Nose', 'Pointy_Nose', 'Big_Lips', 'Smiling', 'Wearing_Lipstick', 'Mouth_Slightly_Open',
                         'Arched_Eyebrows', 'Bags_Under_Eyes', 'Bushy_Eyebrows', 'Narrow_Eyes', 'Eyeglasses', 'Attractive',
                         'Blurry', 'Oval_Face', 'Pale_Skin', 'Young', 'Heavy_Makeup', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 
                         'Gray_Hair', 'Wearing_Earrings', 'Bald', 'Receding_Hairline', 'Bangs', 'Wearing_Hat', 'Straight_Hair', 
                         'Wavy_Hair', '5_o_Clock_Shadow', 'Mustache', 'No_Beard', 'Sideburns', 'Goatee', 'High_Cheekbones',
                         'Rosy_Cheeks', 'Chubby', 'Double_Chin']

        # test mode
        
        self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_MASK_FLIST, config.Attribute_path, mode='test', selected_attrs=selected_attr, augment=False, training=False) #ting
        self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_MASK_FLIST, config.Attribute_path, mode='train', selected_attrs=selected_attr, augment=True, training=True)#ting
        self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_MASK_FLIST, config.Attribute_path, mode='val', selected_attrs=selected_attr, augment=False, training=True)#ting
        self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')       

    def load(self):
        if self.config.MODEL == 1:
            self.attr_model.load()

        elif self.config.MODEL == 2:
            self.inpaint_model.load()

        else:
            self.attr_model.load()
            self.inpaint_model.load()

    def save(self):
        if self.config.MODEL == 1:
            self.attr_model.save()

        elif self.config.MODEL == 2 or self.config.MODEL == 3:
            self.inpaint_model.save()

        else:
            self.attr_model.save()
            self.inpaint_model.save()

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        model = self.config.MODEL
        max_epoch = int(float((self.config.MAX_EPOCH)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return
        
        while(keep_training):
            epoch += 1
            if epoch > max_epoch:
                keep_training = False
                break
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                self.attr_model.train()
                self.inpaint_model.train()

                images, masks, attr = self.cuda(*items) # ting

                mask_percent = self.mask_percent(masks)

                # edge model
                if model == 1:
                    # train
                    outputs, gen_loss, logs = self.attr_model.process(images, masks, attr)
                    logs.append(('mask_percent', mask_percent.item()))
                    logs.append(('predict_attr', outputs[0].cpu().detach().numpy()))
                    logs.append(('groundtruth_attr', attr[0].cpu().detach().numpy()))

                    # backward
                    self.attr_model.backward(gen_loss)
                    iteration = self.attr_model.iteration


                # inpaint model
                elif model == 2:
                    # train
                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, masks, attr)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))
                    logs.append(('mask_percent', mask_percent.item()))

                    # backward
                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration


                # inpaint with attr model
                elif model == 3:
                    # train
                    a_outputs = self.attr_model(images, masks)
               

                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, masks, a_outputs.detach())
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))
                    logs.append(('mask_percent', mask_percent.item()))

                    logs.append(('predict_attr', a_outputs[0].cpu().detach().numpy()))
                    logs.append(('groundtruth_attr', attr[0].cpu().detach().numpy()))

                    # backward
                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration


                # joint model
                else:
                    # train
                    a_outputs, a_gen_loss, a_logs = self.attr_model.process(images, masks, attr)
                    i_outputs, i_gen_loss, i_dis_loss, i_logs = self.inpaint_model.process(images, masks, a_outputs)
                    outputs_merged = (i_outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    i_logs.append(('psnr', psnr.item()))
                    i_logs.append(('mae', mae.item()))
                    logs = a_logs + i_logs
                    logs.append(('mask_percent', mask_percent.item()))

                    # backward
                    self.inpaint_model.backward(i_gen_loss, i_dis_loss)
                    self.attr_model.backward(a_gen_loss)
                    iteration = self.inpaint_model.iteration
                    del outputs_merged, a_outputs, i_outputs

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    self.sample()
            
            self.save() 

        print('\nEnd training....')


    def test(self):
        self.attr_model.eval()
        self.inpaint_model.eval()
        
        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0
        for items in test_loader:
            name = self.test_dataset.load_name(index)
            images, masks, attr = self.cuda(*items)
            index += 1

            mask_percent = self.mask_percent(masks)

            # attr model
            if model == 1:
                outputs, gen_loss, logs = self.attr_model.process(images, masks, attr)
                logs.append(('mask_percent', mask_percent.item()))                   
                logs.append(('predict_attr', outputs[0].cpu().detach().numpy()))
                logs.append(('groundtruth_attr', attr[0].cpu().detach().numpy()))


            # inpaint model
            elif model == 2:
                outputs = self.inpaint_model(images, masks, attr)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs = [('index', index),]
                logs.append(('name', name))
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))
                logs.append(('mask_percent', mask_percent.item()))


            # inpaint with edge model / joint model
            else:
                attrs_predict = self.attr_model(images, masks).detach()
                outputs = self.inpaint_model(images, masks, attrs_predict)
                outputs_merged = (outputs * masks) + (images * (1 - masks))
                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs = [('index', index),]
                logs.append(('name', name))
                logs.append(('predict_attr', attrs_predict[0].cpu().detach().numpy()))
                logs.append(('groundtruth_attr', attr[0].cpu().detach().numpy()))
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))
                logs.append(('mask_percent', mask_percent.item()))
                
            path = os.path.join(self.results_path, name)
            
            
            if model != 1:
                outputs_merged = self.postprocess(outputs_merged)[0]
                outputs = self.postprocess(outputs)[0]
                print(index, name)
                fname, fext = name.split('.')
                imsave(outputs_merged, path)
                
                if self.debug:
                    masked = self.postprocess(images * (1 - masks) + masks)[0]
                    imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))

        
            test_file = os.path.join(self.results_path, 'log_' + 'test' + '.dat')

            with open(test_file, 'a') as f:
                f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))
 
        print('\nEnd test....')


    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.attr_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL
        items = next(self.sample_iterator)
        images, masks, attr = self.cuda(*items)

        mask_percent = self.mask_percent(masks)

        # attr model
        if model == 1:
            iteration = self.attr_model.iteration
            inputs = (images * (1 - masks)) + masks
            outputs, _, a_logs = self.attr_model.process(images, masks, attr)
            a_logs.append(('mask_percent', mask_percent.item()))

            for i in range(outputs.shape[0]):
                a_logs.append(('predict_attr', outputs[i].cpu().detach().numpy()))
                a_logs.append(('groundtruth_attr', attr[i].cpu().detach().numpy()))


        # inpaint model
        elif model == 2:
            iteration = self.inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            outputs = self.inpaint_model(images, masks, attr)
            outputs_merged = (outputs * masks) + (images * (1 - masks))


        # inpaint with edge model / joint model
        else:
            iteration = self.inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            a_outputs, _, a_logs = self.attr_model.process(images, masks, attr)
            outputs = self.inpaint_model(images, masks, a_outputs)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

            a_logs.append(('mask_percent', mask_percent.item()))
            # metrics
            psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
            mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
            a_logs.append(('psnr', psnr.item()))
            a_logs.append(('mae', mae.item()))

            for i in range(a_outputs.shape[0]):
                a_logs.append(('predict_attr', a_outputs[i].cpu().detach().numpy()))
                a_logs.append(('groundtruth_attr', attr[i].cpu().detach().numpy()))

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1


        path = os.path.join(self.samples_path, self.model_name)
        create_dir(path)
        if model != 1:
            images = stitch_images(
                self.postprocess(images),
                self.postprocess(inputs),
                self.postprocess(outputs),
                self.postprocess(outputs_merged),
                img_per_row = image_per_row
            )

            name = os.path.join(path, str(iteration).zfill(5) + ".png")
            print('\nsaving sample ' + name)
            images.save(name)
            
        if model != 2:
            attr_file = os.path.join(path, 'log_' + 'attr' + '.dat')
            a_logs = [("it", iteration), ] + a_logs
            with open(attr_file, 'a') as f:
                f.write('%s\n' % ' '.join([str(item[1]) for item in a_logs]))

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def mask_percent(self, mask):
        holes = torch.sum((mask > 0).float())
        pixel_num = torch.sum((mask >= 0).float())
        percent = holes/pixel_num
        return percent