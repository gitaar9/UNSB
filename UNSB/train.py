import time
import torch
from matplotlib import pyplot as plt

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util import util
import wandb


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset2 = create_dataset(opt)
    dataset_size = len(dataset)    # get the number of images in the dataset.

    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    wandb_run = wandb.init(project="UNSB", name="test") # Initialize a new run
    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1

    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        dataset.set_epoch(epoch)
        for i, (data, data2) in enumerate(zip(dataset, dataset2)):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data,data2)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data,data2)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                # Print to std out
                message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, epoch_iter, optimize_time, t_data)
                for k, v in losses.items():
                    message += '%s: %.3f ' % (k, v)
                print(message)  # print the message

                # Log to wandb
                log_dict = {
                    'epoch': epoch,
                    "losses": {k: v for k, v in losses.items()},
                    'lr': model.optimizers[0].param_groups[0]['lr']
                }
                model.compute_visuals()
                visuals = {label: util.tensor2im(tensor) for label, tensor in model.get_current_visuals().items()}
                fig, axs = plt.subplots(1, 3, figsize=(9, 3))
                axs[0].imshow(visuals['real_A'], cmap='gray')
                axs[0].set_title('Real A')
                axs[0].axis('off')
                axs[1].imshow(visuals['real_A_noisy'], cmap='gray')
                axs[1].set_title('Real A Noisy')
                axs[1].axis('off')
                axs[2].imshow(visuals['fake_B'], cmap='gray')
                axs[2].set_title('Fake B')
                axs[2].axis('off')
                plt.tight_layout()
                log_dict["real2fake"] = fig
                plt.close()

                # Plot for real_B and idt_B
                fig, axs = plt.subplots(1, 2, figsize=(6, 3))
                axs[0].imshow(visuals['real_B'], cmap='gray')
                axs[0].set_title('Real B')
                axs[0].axis('off')
                axs[1].imshow(visuals['idt_B'], cmap='gray')
                axs[1].set_title('Idt B')
                axs[1].axis('off')
                plt.tight_layout()
                log_dict["fake2fake"] = fig
                plt.close()

                wandb_run.log(log_dict)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
