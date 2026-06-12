"""Main Training script for the STE-GAN.

Adapted from: https://github.com/descriptinc/cargan/blob/master/cargan/train.py
"""
# import os #  - Adaptation for running locally
# # Change to a new directory path
# if not(os.getcwd().endswith('EMGAN')):
#     os.chdir('projeto/dgm-2026.1/projects/EMGAN')

import argparse
import functools
import itertools
import logging
import random
import sys
sys.path.insert(0, "./NeuroRVQ")
sys.path.append(".")
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

import ste_gan
from ste_gan.constants import DataType
from ste_gan.data.emg_dataset import EMGDataset
from ste_gan.data.loader import loaders_via_config
from ste_gan.utils.plot_utils import plot_real_vs_fake_emg_signal_with_envelope
from ste_gan.losses.emg_encoder_loss import (EMGEncoderLoss,
                                             EMGEncoderLossOutput)
from ste_gan.losses.time_domain_loss import MultiTimeDomainFeatureLoss
from ste_gan.losses.neuro_rvq_loss import NeuroRVQLoss
from ste_gan.models.discriminator import init_emg_discriminators
from ste_gan.models.emg_encoder import load_emg_encoder
from ste_gan.models.generator import init_emg_generator
from ste_gan.train_utils import (add_eval_hyperparams_to_parser,
                                 create_ste_gan_model_name, load_config,
                                 mean_error, phoneme_accuracy,
                                 phoneme_accuracy_no_silence)
from ste_gan.utils.common import load_latest_checkpoint

def create_tensorboard(model_directory):
    logging.info(f"Writing tensorboard logs to: {model_directory}")
    writer = SummaryWriter(str(model_directory.absolute()))
    return writer

def create_scheduler(obj):

    scheduler_fn = functools.partial(
        torch.optim.lr_scheduler.ExponentialLR,
        gamma=.999,
        last_epoch=obj.start_epoch if obj.checkpoint is not None else -1
    )

    return scheduler_fn(obj.optG), scheduler_fn(obj.optD)

def load_checkpoint(obj):
    if obj.checkpoint is not None:
        logging.info(f"Loading checkpoint: {obj.checkpoint}")
        return load_latest_checkpoint(obj.checkpoint, obj.device, obj.netG, obj.netD, obj.optG, obj.optD)
    else:
        return obj.netG, obj.netD, obj.optG, obj.optD, -1, 0

def compile(obj): # compiling models and losses

    if int(torch. __version__[0]) >= 2:
        logging.info(f"Compiling models...PyTorch version: {torch.__version__}")
        #multi_td_loss = torch.compile(multi_td_loss)
        return torch.compile(obj.netG), torch.compile(obj.netD), torch.compile(obj.emg_encoder), torch.compile(obj.emg_encoder_loss)
    else:
        logging.warning(f"Will NOT compile models. Torch version: {torch. __version__}")
        return obj.netG, obj.netD, obj.emg_encoder, obj.emg_encoder_loss
    
def our_logging(obj, epoch, iterno, loss_G, log_start): # logs for training and validation results

    if obj.mode == 'train':
        phoneme_acc_train = phoneme_accuracy(obj.train_num_phones, obj.train_num_phones_correct)
        phoneme_acc_train_no_sil = phoneme_accuracy_no_silence(obj.train_num_phones, obj.train_num_phones_correct_no_silence, obj.train_num_silence)
        
        log = (
            f"Epoch {epoch} ({iterno}/{len(obj.dataloader.train_loader)}) | Steps {obj.steps} | "
            f"ms/batch {1e3 * (time.time() - log_start) / obj.cfg.train.interval_log:5.2f} | Train Loss: {loss_G.item():.4f} | Ph. Acc. (Avg.) {phoneme_acc_train:.2f} | Ph. Acc. Avg. (No Sil) {phoneme_acc_train_no_sil:.2f}"
        )
        obj.writer.add_scalar("train_loss/phoneme_accuracy_avg", phoneme_acc_train, obj.steps)
        obj.writer.add_scalar("train_loss/phoneme_accuracy_avg_no_sil", phoneme_acc_train_no_sil, obj.steps)
        logging.info(log)

    else: # validation logging

        avg_val_td_error = mean_error(obj.td_errors)
        avg_val_phoneme_error = mean_error(obj.phoneme_errors)
        avg_val_wave_error = mean_error(obj.wave_errors)
        avg_su_error = mean_error(obj.su_errors)
        avg_phoneme_accuracy = phoneme_accuracy(obj.val_num_phones, obj.val_num_phones_correct)
        avg_phoneme_accuracy_no_sil = phoneme_accuracy_no_silence(obj.val_num_phones, 
                                                                obj.val_num_phones_correct_no_silence,
                                                                obj.val_num_silence)
    # Log validation errors to tensorboard
        obj.writer.add_scalar("val_loss/speech_unit", avg_su_error, obj.steps)
        obj.writer.add_scalar("val_loss/multi_td", avg_val_td_error, obj.steps)
        obj.writer.add_scalar("val_loss/phoneme", avg_val_phoneme_error, obj.steps)
        obj.writer.add_scalar("val_loss/phoneme_accuracy_avg", avg_phoneme_accuracy, obj.steps)
        obj.writer.add_scalar("val_loss/phoneme_accuracy_avg_no_sil", avg_phoneme_accuracy_no_sil, obj.steps)
        obj.writer.add_scalar("val_loss/waveform", avg_val_wave_error, obj.steps)
        
        logging.info("-" * 100)
        logging.info("Took %5.4fs to run validation loop" % (time.time() - log_start))
        if obj.cfg.train.loss_neuro_rvq_error and obj.rvq_errors:
            avg_val_rvq_error = mean_error(obj.rvq_errors)
            obj.writer.add_scalar("val_loss/neuro_rvq", avg_val_rvq_error, obj.steps)
            logging.info(f"\t - Avg. Val. NeuroRVQ Error: {avg_val_rvq_error}")
        logging.info(f"\t - Avg. Val. Speech Unit Error : {avg_su_error}")
        logging.info(f"\t - Avg. Val. Multi-TD Val. Error: {avg_val_td_error}")
        logging.info(f"\t - Avg. Val. Phoneme Error: {avg_val_phoneme_error}")
        logging.info(f"\t - Avg. Val. Phoneme Accuracy: {avg_phoneme_accuracy}")
        logging.info(f"\t - Avg. Val. Phoneme Accuracy (No Sil.): {avg_phoneme_accuracy_no_sil}")
        logging.info(f"\t - Avg. Val. Waveform Error: {avg_val_wave_error}")
        logging.info("-" * 100)
    
        if avg_su_error < obj.best_su_loss: # saving best model based on su loss
            obj.best_su_loss = avg_su_error
            logging.info(f"Saving best model with best val. SU error @ {obj.best_su_loss:5.4f}...")
            torch.save(obj.netG.state_dict(), obj.model_directory / "best_netG.pt")
            torch.save(obj.netD.state_dict(), obj.model_directory / "best_netD.pt")

        logging.info("-" * 100)
        logging.info("Took %5.4fs to run validation loop" % (time.time() - log_start))
        logging.info("-" * 100)

def model_save(obj, epoch, mode='intermediate'): # saves models parameters

    save_start = time.time()
    logging.info("Starting to save models...")

    dict = {
            'epoch': epoch,
            'steps': obj.steps,
            'optG': obj.optG.state_dict(),
            'optD': obj.optD.state_dict(),
        }

    if mode == 'intermediate':
        generator_pt, discriminator_pt, opt_path = f'netG-{obj.steps:08d}.pt', f'netD-{obj.steps:08d}.pt', f'checkpoint-{obj.steps:08d}.pt'

    elif mode == 'final':
        logging.info(f"Finished training script. Starting saving last model")
        generator_pt, discriminator_pt, opt_path = f'netG-final.pt', f'netD-final.pt', f'checkpoint-final.pt'

        logging.info(f"Writing a .done file in {obj.model_directory}")
        with open(obj.model_directory / ".done", '+w') as fp:
            fp.write(f"done: {time.time()}")
    
    elif mode == 'epoch':
        logging.info(f"Finished training epoch {epoch}. Starting saving last model")
        generator_pt, discriminator_pt, opt_path = f'netG-final.pt', f'netD-final.pt', f'checkpoint-final.pt'

    torch.save( # saving generator
        obj.netG.state_dict(),
        obj.model_directory / generator_pt)
    torch.save( # saving discriminator
        obj.netD.state_dict(),
        obj.model_directory / discriminator_pt) 
    # saving optimizers
    torch.save(dict, obj.model_directory / opt_path)
    
    # logging saving time
    logging.info('-' * 100)
    logging.info('Took %5.4fs to save checkpoint' % (time.time() - save_start))
    logging.info('-' * 100)

class our_dataloader():

    def __init__(self, cfg, model_directory):

        logging.info("Loading Data -- this can take a while")
        data_root = Path(cfg.data.dataset_root)
        logging.info(f"Data Set root: {data_root}")

        # creates dataloaders for training
        self.train_loader, self.valid_loader, self.test_loader = loaders_via_config(cfg)

        # Save Session and Speaking Mode ID mappings
        train_data_set: EMGDataset = self.train_loader.dataset
        train_data_set.save_session_and_speaking_mode_mapping_json(model_directory)

    def data_from_dict(self, obj, dict, speech_feature_type): # extracts all input data orderly

        # common data
        x_t = dict[DataType.REAL_EMG].to(obj.device)
        spk_mode_idx = dict[DataType.SPEAKING_MODE_INDEX].to(obj.device)
        sess_idx = dict[DataType.SESSION_INDEX].to(obj.device)
        phoneme_targets = dict[DataType.PHONEMES].to(obj.device)

        # speech units
        speech_units_t = dict[DataType.SPEECH_UNITS].to(obj.device) # speech unit for loss calculation

        # speech embedding to serve as generator input
        if speech_feature_type == DataType.SPEECH_UNITS:
            s_t1 = speech_units_t
        else:    
            s_t1 = dict[speech_feature_type].to(obj.device)

        return x_t, spk_mode_idx, sess_idx, phoneme_targets, speech_units_t, s_t1

class trainer():

    def __init__(self, cfg: DictConfig,
                        model_directory: Path, 
                        checkpoint: Path, 
                        torch_device: str,
                        debug: bool,
                        emg_enc_ckpt: Path = None):
        
        # setting seeds
        np.random.seed(cfg.train.random_seed)
        torch.cuda.manual_seed(cfg.train.random_seed)
        torch.manual_seed(cfg.train.random_seed)
        random.seed(cfg.train.random_seed)

        # instantiated variables
        self.cfg = cfg
        self.model_directory = model_directory
        self.checkpoint = checkpoint
        self.torch_device = torch_device
        self.debug = debug
        self.emg_enc_ckpt = emg_enc_ckpt
        self.mode = None # train or valid
 
        # hyperparameters
        self.device = torch.device(self.torch_device)

        # models - pre-trained
        logging.info(f"Initializing EMG Encoder Model with EMG encoder checkpoint: {self.emg_enc_ckpt}")
        self.emg_encoder = self.freeze_parameters(load_emg_encoder(self.cfg, self.device, self.emg_enc_ckpt)).to(self.device)
        self.emg_encoder.eval() # change behavior of Dropout and BN

        # models - to be trained
        logging.info(f"Initializing Models")
        self.netG = init_emg_generator(self.cfg).to(self.device)
        self.netD = init_emg_discriminators(self.cfg).to(self.device)

        # losses
        logging.info(f"Initializing Losses")
        self.multi_td_loss = MultiTimeDomainFeatureLoss(self.cfg.data.num_emg_channels).to(self.device)
        self.emg_encoder_loss = EMGEncoderLoss(self.emg_encoder).to(self.device)
        if self.cfg.train.loss_neuro_rvq_error: 
            self.neuro_rvq_loss = NeuroRVQLoss(self.cfg).to(self.device)
        # optims
        self.optG = ste_gan.OPTIMIZER(self.netG.parameters())
        self.optD = ste_gan.OPTIMIZER(self.netD.parameters())

        # Mixed Precision Scaler    
        self.scaler = torch.cuda.amp.GradScaler()

        # tensorboard
        self.writer = create_tensorboard(self.model_directory)

        # recover variables from checkpoint
        self.netG, self.netD, self.optG, self.optD, self.start_epoch, self.steps = \
            load_checkpoint(self)
        
        # optims schedulers
        self.scheduler_g, self.scheduler_d = create_scheduler(self)

        # compile models
        self.netG, self.netD, self.emg_encoder, self.emg_encoder_loss = compile(self)

        # variables for storing losses
        self.td_errors = []
        self.su_errors = []
        self.phoneme_errors = []
        self.wave_errors = []
        self.rvq_errors = []
        self.best_td_loss = np.inf
        self.best_su_loss = np.inf # currently used for best model

        # variables for storing metrics
        self.train_num_phones = 0
        self.train_num_phones_correct = 0
        self.train_num_silence = 0
        self.train_num_phones_correct_no_silence = 0

        self.val_num_phones = 0
        self.val_num_phones_correct = 0
        self.val_num_silence = 0
        self.val_num_phones_correct_no_silence = 0

    def freeze_parameters(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model
    
    def unfreeze_parameters(self, model):
        for param in model.parameters():
            param.requires_grad = True
        return model
    
    def extract_metrics(self, emg_enc_loss_output): # calculates metrics from training and validation

        # calculating phoneme accuracy including silences
        phoneme_acc_batch = phoneme_accuracy(emg_enc_loss_output.num_phones, emg_enc_loss_output.num_correct_phones)

        # calculating phoneme accuracy ignoring silences
        phoneme_acc_batch_no_silence = phoneme_accuracy_no_silence(
            emg_enc_loss_output.num_phones, 
            emg_enc_loss_output.num_correct_phones_no_silence,
            emg_enc_loss_output.num_silence_phones)

        if self.mode == 'train':
            self.writer.add_scalar("train_loss/phoneme_acc_batch", phoneme_acc_batch, self.steps)
            self.writer.add_scalar("train_loss/phoneme_acc_batch_no_sil", phoneme_acc_batch_no_silence, self.steps)
            
            # adding global metrics
            self.train_num_phones += emg_enc_loss_output.num_phones
            self.train_num_phones_correct += emg_enc_loss_output.num_correct_phones
            self.train_num_phones_correct_no_silence += emg_enc_loss_output.num_correct_phones_no_silence
            self.train_num_silence += emg_enc_loss_output.num_silence_phones

        else:
            self.val_num_phones += emg_enc_loss_output.num_phones
            self.val_num_phones_correct += emg_enc_loss_output.num_correct_phones
            self.val_num_silence += emg_enc_loss_output.num_silence_phones
            self.val_num_phones_correct_no_silence += emg_enc_loss_output.num_correct_phones_no_silence
    
    def d_training_loop(self, x_t, x_pred_t): # discriminator training inside loop

        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.cfg.train.mixed_precision):
            # if using this loss
            if self.cfg.train.loss_adversarial:
                D_fake_det = self.netD(x_pred_t.detach())
                D_real = self.netD(x_t)
                loss_D = 0
                # adding the losses (autograd knows which portion of the loss belongs to each network)
                for scale in D_fake_det: # for each discriminator
                    loss_D += F.mse_loss(scale[-1], torch.zeros_like(scale[-1])) # for the last layer
                for scale in D_real:
                    loss_D += F.mse_loss(scale[-1], torch.ones_like(scale[-1]))
                # the discriminator returns a list because of the feature loss (per layer) to be used (for the generator)
                self.writer.add_scalar("train_loss/discriminator", loss_D.item(), self.steps)

        self.scaler.scale(loss_D).backward()
        self.scaler.step(self.optD) # only updates discriminator

    def g_training_loop(self, x_t, x_pred_t, speech_units_t, phoneme_targets): # generator training inside loop
        
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.cfg.train.mixed_precision):
            self.netD = self.freeze_parameters(self.netD) # freezing discriminator parameters to save memory
            D_fake = self.netD(x_pred_t)
            D_real = self.netD(x_t)
            loss_G = 0 # defining Generator loss

            if self.cfg.train.loss_adversarial: # inverted discriminator loss
                for scale in D_fake:
                    loss_G += F.mse_loss(scale[-1], torch.ones_like(scale[-1]))

            # Feature matching loss
            if self.cfg.train.loss_feat_match_error:
                loss_feat = 0
                for i in range(len(D_fake)): # each discriminator
                    for j in range(len(D_fake[i]) - 1): # each layer except the last one, since we want features
                        loss_feat += F.l1_loss(D_fake[i][j], D_real[i][j].detach()) # calculate L1 loss for specific layer
                loss_G += self.cfg.train.loss_feat_match_weight * loss_feat # multiplying by a lambda (weight)
                self.writer.add_scalar("train_loss/feature_matching", loss_feat.item(), self.steps)

            # L1 error on multi-time-domain features (between fake and real signals)
            if self.cfg.train.loss_multi_td_error:
                td_error = self.multi_td_loss(x_t, x_pred_t)
                loss_G += self.cfg.train.loss_multi_td_weight * td_error # multiplying by a lambda (weight)
                self.writer.add_scalar("train_loss/multi_td", td_error.item(), self.steps)

            # MSE error on waveform (signal)
            if self.cfg.train.loss_waveform_error:
                wave_loss = torch.nn.functional.mse_loss(x_pred_t, x_t)
                loss_G += self.cfg.train.loss_waveform_weight * wave_loss # multiplying by a lambda (weight)
                self.writer.add_scalar("train_loss/waveform", wave_loss.item(), self.steps)

            # EMG encoder (frozen) losses -> speech unit distance and phoneme cross entropy
            # forward step regardless since we are extracting metrics
            emg_enc_loss_output: EMGEncoderLossOutput = self.emg_encoder_loss(x_pred_t, speech_units_t, phoneme_targets)

            # extract metrics - from EMG encoder output
            self.extract_metrics(emg_enc_loss_output)

            # speech unit loss
            if self.cfg.train.loss_speech_unit_error:
                su_loss = emg_enc_loss_output.speech_unit_loss
                loss_G += self.cfg.train.loss_speech_unit_weight * su_loss # multiplying by a lambda (weight)
                self.writer.add_scalar("train_loss/speech_unit", su_loss.item(), self.steps)
            
            # phoneme loss
            if self.cfg.train.loss_phoneme_error:
                phoneme_loss = emg_enc_loss_output.phoneme_loss
                loss_G += self.cfg.train.loss_phoneme_weight * phoneme_loss # multiplying by a lambda (weight)
                self.writer.add_scalar("train_loss/phoneme", phoneme_loss.item(), self.steps)
            # neurorvq loss
            if self.cfg.train.loss_neuro_rvq_error:
                rvq_loss = self.neuro_rvq_loss(x_t, x_pred_t)
                loss_G += self.cfg.train.loss_neuro_rvq_weight * rvq_loss
                self.writer.add_scalar("train_loss/neuro_rvq", rvq_loss.item(), self.steps)
            # logging total generator loss
            self.writer.add_scalar("train_loss/generator", loss_G.item(), self.steps)

        self.scaler.scale(loss_G).backward()
        self.scaler.step(self.optG) # only updates generator
        self.scaler.update(),  # updating scaler after training step  per iteration, after both D and G steps


        self.netD = self.unfreeze_parameters(self.netD) # unfreezing discriminator parameters for further training

        return loss_G # for logging
    
    def validation(self, speech_feature_type): # validation losses and metrics
        # our_logging sees only the current validation pass, not accumulated history
        self.td_errors = []
        self.su_errors = []
        self.phoneme_errors = []
        self.wave_errors = []
        self.rvq_errors = []
        self.val_num_phones = 0
        self.val_num_phones_correct = 0
        self.val_num_silence = 0
        self.val_num_phones_correct_no_silence = 0
        logging.info(f"Starting validation")

        # will not forward pass on discriminator (not interested in its losses)
        self.netG.eval() # for Dropout and BN
        self.mode = 'valid' # for metrics
        
        with torch.no_grad():
            # Batch loop - Validation
            for i, val_dict in enumerate(self.dataloader.valid_loader):
                        
                x_t, spk_mode_idx, sess_idx, phoneme_targets, speech_units_t, s_t1 = \
                    self.dataloader.data_from_dict(self, val_dict, speech_feature_type)
                
                # calculation of metrics and main losses, based on generation
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    x_pred_t = self.netG(s_t1, sess_idx, spk_mode_idx) # generator inference

                    # validation loss calculations
                    self.wave_errors.append(torch.nn.functional.mse_loss(x_pred_t, x_t).item())
                    self.td_errors.append(self.multi_td_loss(x_t, x_pred_t).item())
                    emg_enc_loss_output = self.emg_encoder_loss(x_pred_t, speech_units_t, phoneme_targets)
                
                    self.su_errors.append(emg_enc_loss_output.speech_unit_loss.item())
                    self.phoneme_errors.append(emg_enc_loss_output.phoneme_loss.item())
                    if self.cfg.train.loss_neuro_rvq_error:  
                        self.rvq_errors.append(self.neuro_rvq_loss(x_t, x_pred_t).item())
                    # metrics calculations
                    self.extract_metrics(emg_enc_loss_output)

    def generation(self, speech_feature_type): # generates sample from validation set

        logging.info("Starting to generate validation samples for plotting")
        gen_start = time.time()

        self.netG.eval()
        self.mode = 'valid'

        with torch.no_grad():
            for i, sample_dict in enumerate(self.dataloader.valid_loader.dataset):
            
                _, spk_mode_idx, sess_idx, _, _, s_t1 = \
                    self.dataloader.data_from_dict(self, sample_dict, speech_feature_type)
                                
                # Generate the real EMG signal
                real_emg = sample_dict[DataType.REAL_EMG].squeeze(0).detach().cpu().numpy()
                
                # Generate the fake EMG signal
                pred_emg = self.netG.generate(s_t1.unsqueeze(0), sess_idx.unsqueeze(0), spk_mode_idx.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
                
                plot_real_vs_fake_emg_signal_with_envelope(
                    real_emg_signal=real_emg,
                    fake_emg_signal=pred_emg,
                    file_id=f"Validation sample {i}",
                    save_as=None,
                    tb_summary_writer=self.writer,
                    tb_tag_prefix="val/envelopes_emg_real_vs_fake",
                    global_step=self.steps,
                    show=False
                )

                if i > self.cfg.train.num_test_samples:
                    break
            
        logging.info("-" * 100)
        logging.info("Took %5.4fs to generate samples" % (time.time() - gen_start))
        logging.info("-" * 100)
    
    def train(self): # main loop

        logging.info(f"Starting Training")
        log_start = time.time()
        
        # enable cudnn autotuner to speed up training
        torch.backends.cudnn.benchmark = True

        # capturing data
        self.dataloader = our_dataloader(self.cfg, self.model_directory)
        
        # speech unit default
        speech_feature_type = self.cfg.model.speech_feature_type # SPEECH_UNITS default

        # Epoch loop
        for epoch in itertools.count(self.start_epoch):
            logging.info(f"Starting epoch {epoch+1}")
            epoch_start_time = time.time()

	
            # reflects only the current epoch, not the entire training history
            self.train_num_phones = 0
            self.train_num_phones_correct = 0
            self.train_num_silence = 0
            self.train_num_phones_correct_no_silence = 0

            # Batch loop - Training
            for iterno, train_dict in enumerate(self.dataloader.train_loader):
                # extracting all inputs
                x_t, spk_mode_idx, sess_idx, phoneme_targets, speech_units_t, s_t1 = \
                    self.dataloader.data_from_dict(self, train_dict, speech_feature_type)

                # resseting grad
                self.optD.zero_grad()
                self.optG.zero_grad()

                # training mode
                self.netG.train()
                self.netD.train()
                self.mode = 'train' # for metrics

                # generator forward pass, for both training steps
                x_pred_t = self.netG(s_t1, sess_idx, spk_mode_idx)

                # the discriminator training loop (adversarial loss)
                self.d_training_loop(x_t, x_pred_t)

                # the generator training loop (all losses, including NeuroRVQ)
                loss_G = self.g_training_loop(x_t, x_pred_t, speech_units_t, phoneme_targets)

                # Logging of losses and metrics up to now
                if self.steps % self.cfg.train.interval_log == 0:
                    our_logging(self, epoch, iterno, loss_G, log_start)
                    log_start = time.time()  # reset timer after each log 
                # Validation
                if self.steps % self.cfg.train.interval_valid == 0:
                    val_start = time.time()
                    self.validation(speech_feature_type)    
                    # Calculate Mean Validation errors
                    our_logging(self, None, None, None, val_start) # logging for validation results
                
                # Generation
                if (self.steps % self.cfg.train.interval_sample == 0):
                    self.generation(speech_feature_type)

                # Save checkpoint
                if (self.steps > 0) and (self.steps % self.cfg.train.interval_save == 0):
                    model_save(self, epoch, 'intermediate')

                # Save model at last training step 
                if self.steps >= self.cfg.train.max_steps:
                    model_save(self, epoch, 'final')
                    return
                            
                self.steps += 1

            # optim scheduler steps after each epoch
            if self.cfg.train.loss_adversarial:
                self.scheduler_d.step()
            self.scheduler_g.step()
            
            # log epoch training time
            epoch_end_time = time.time()
            logging.info(f"Finished training epoch {epoch}. Elapsed time: {epoch_end_time - epoch_start_time}")
            
            # Save "last" model every 5 epochs
            if epoch % 5 == 0:
                model_save(self, epoch, 'epoch')

        return 1

###############################################################################
# Entry point
###############################################################################
def main(cfg: DictConfig, continue_run: bool, debug: bool, emg_enc_ckpt: Path, **kwargs):
    dataset_root = cfg.data.dataset_root
    print(f"Data root: {dataset_root}")
    print(f"continue_run: {continue_run}")
    print(f"Debug (argparse): {debug}")
    
    if not debug and cfg.train.debug:
        print(f"WARNING: SETTING GLOBAL DEBUG FLAG")
        debug = True
    
    # Create output dir
    model_base_dir = Path(cfg.model_base_dir)
    output_directory = model_base_dir/ create_ste_gan_model_name(
        cfg, add_timestamp=False, debug=debug,
    )
    if output_directory.exists() and continue_run:
        logging.info(f"WARNING: Removing old model directory: {output_directory}")
        checkpoint = output_directory
    else:
        checkpoint = None
    output_directory.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_directory}")

    done_file = output_directory / ".done"
    if output_directory and done_file.exists():
        logging.warning(f"Exiting training script as '.done' file exists: {done_file.absolute()}")
        sys.exit()

    # Save configuration
    config_file = output_directory / "config.yaml"
    if not config_file.exists():
        with open(config_file, '+w') as fp:
            OmegaConf.save(config=cfg, f=fp.name)

    logging.info(OmegaConf.to_yaml(cfg))
    logging.getLogger().setLevel(logging.INFO)
    log_file = output_directory / "log.txt"
    fh = logging.FileHandler(str(log_file.absolute()))
    logging.getLogger().addHandler(fh) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_class = trainer(cfg, output_directory, checkpoint, device, debug, emg_enc_ckpt)
    train_class.train()

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ste_gan_base_gantts.yaml",
                        help="The main training configuration for this run.")
    parser.add_argument("--data", type=str, default="configs/data/gaddy_and_klein_corpus.yaml",
                        help="A path to a data configuration file.")
    parser.add_argument("--emg_enc_cfg", type=str, default="configs/emg_encoder/conv_transformer.yaml",
                        help="A path to an EMG encoder configuration file.")
    parser.add_argument("--emg_enc_ckpt", type=str, default="exp/emg_encoder/EMGEncoderTransformer_voiced_only__seq_len__200__data_gaddy_complete/best_val_loss_model.pt",
                        help="A path to a checkpooint of a pre-trained EMG encoder. Must correspond to the EMG encoder configuration in 'emg_enc_cfg'.")
    parser.add_argument(
        '--checkpoint',
        type=Path,
        help='Optional checkpoint to start training from')
    parser.add_argument(
        '--continue_run',
        action='store_true',
        help='Whether to continue training')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Whether to run the training script in debug mode.')
    
    parser = add_eval_hyperparams_to_parser(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)
    args.cfg = cfg
    main(**vars(args))
