import torch
import config
import utils
from dataset import JitteredDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm 
from discriminator import Discriminator
from generatorUnet import Generator

def train_function(
    disc_jitter, disc_unjitter, gen_jitter, gen_unjitter, train_loader,
    val_loader, opt_disc, opt_gen, L1, MSE, disc_scaler, gen_scaler, epoch, 
    ):

    loop = tqdm(train_loader, leave=True)
     
    for _, (jittered, unjittered) in enumerate(loop):
        jittered = jittered.to(config.DEVICE)
        unjittered = unjittered.to(config.DEVICE)

        # Train discriminator jittered and unjittered 
        with torch.cuda.amp.autocast():
            # Find disc loss for unjittered 
            # Generate fake unjittered image
            fake_unjittered = gen_unjitter(jittered)
            # Get disc score for real and fake unjittered image
            disc_unjittered_real = disc_unjitter(unjittered)
            # Use detach and this will be used to train generator aswell
            disc_unjittered_fake = disc_unjitter(fake_unjittered.detach())
            
            disc_unjittered_real_loss = MSE(disc_unjittered_real, torch.ones_like(disc_unjittered_real))
            disc_unjittered_fake_loss = MSE(disc_unjittered_fake, torch.zeros_like(disc_unjittered_fake))

            disc_unjittered_loss = disc_unjittered_real_loss + disc_unjittered_fake_loss

            # Find disc loss for jittered 
            # Generate fake jittered image
            fake_jittered = gen_jitter(unjittered)
            # Get disc score for real and fake jittered image
            disc_jittered_real = disc_jitter(jittered)
            # Use detach and this will be used to train generator aswell
            disc_jittered_fake = disc_jitter(fake_jittered.detach())
            
            disc_jittered_real_loss = MSE(disc_jittered_real, torch.ones_like(disc_jittered_real))
            disc_jittered_fake_loss = MSE(disc_jittered_fake, torch.zeros_like(disc_jittered_fake))

            disc_jittered_loss = disc_jittered_real_loss + disc_jittered_fake_loss

            # Total discriminator loss
            disc_loss = (disc_unjittered_loss + disc_jittered_loss)/2

        # Compute backpropogation for generators 
        opt_disc.zero_grad()
        disc_scaler.scale(disc_loss).backward()
        disc_scaler.step(opt_disc)
        disc_scaler.update()

        # Train generator jittered and unjittered 
        with torch.cuda.amp.autocast():
            # Adversarial loss for both generators
            disc_unjittered_fake = disc_unjitter(fake_unjittered)
            disc_jittered_fake = disc_jitter(fake_jittered)
            
            gen_jittered_loss = MSE(disc_jittered_fake, torch.ones_like(disc_jittered_fake))
            gen_unjittered_loss = MSE(disc_unjittered_fake, torch.ones_like(disc_unjittered_fake))

            # Cycle loss
            cycle_jittered = gen_jitter(fake_unjittered)
            cycle_unjittered = gen_unjitter(fake_jittered)

            cycle_jittered_loss = L1(jittered, cycle_jittered)
            cycle_unjittered_loss = L1(unjittered, cycle_unjittered)
            
            # Identity loss
            identity_jittered = gen_jitter(jittered)
            identity_unjittered = gen_unjitter(unjittered)

            identity_jittered_loss = L1(jittered, identity_jittered)
            identity_unjittered_loss = L1(unjittered, identity_unjittered)

            # Total generator loss
            gen_loss = (
                gen_unjittered_loss + 
                gen_jittered_loss +
                cycle_jittered_loss * config.LAMBDA_CYCLE +
                cycle_unjittered_loss * config.LAMBDA_CYCLE + 
                identity_jittered_loss * config.LAMBDA_IDENTITY + 
                identity_unjittered_loss * config.LAMBDA_IDENTITY
            )
        # Compute backpropogation for generators
        opt_gen.zero_grad()
        gen_scaler.scale(gen_loss).backward()
        gen_scaler.step(opt_gen)
        gen_scaler.update()

    if epoch % 5 ==0:
        utils.save_unjittered_examples(gen_unjitter, val_loader, epoch, config.TRAIN_IMAGE_FILE)

def main():
    disc_jitter = Discriminator(inChannels=config.CHANNELS_IMG).to(config.DEVICE) # for horse
    disc_unjitter = Discriminator(inChannels=config.CHANNELS_IMG).to(config.DEVICE) # for zebra
    
    gen_jitter = Generator(
        imageChannels=config.CHANNELS_IMG,
        numResiduals=config.NUM_RESIDUALS,
    ).to(config.DEVICE)
    gen_unjitter = Generator(
        imageChannels=config.CHANNELS_IMG,
        numResiduals=config.NUM_RESIDUALS,
    ).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_jitter.parameters()) + list(disc_unjitter.parameters()),
        lr = config.LEARNING_RATE,
        betas = config.OPTIMISER_WEIGHTS
    )
    opt_gen = optim.Adam(
        list(gen_jitter.parameters()) + list(gen_unjitter.parameters()),
        lr = config.LEARNING_RATE,
        betas = config.OPTIMISER_WEIGHTS
    )

    L1 = nn.L1Loss()
    MSE = nn.MSELoss()

    if config.LOAD_MODEL:
        utils.load_checkpoint(
            config.CHECKPOINT_DISC_JITTER_LOAD, disc_jitter, opt_disc, config.LEARNING_RATE,
        )
        utils.load_checkpoint(
            config.CHECKPOINT_DISC_UNJITTER_LOAD, disc_unjitter, opt_disc, config.LEARNING_RATE,
        )

        utils.load_checkpoint(
            config.CHECKPOINT_GEN_JITTER_LOAD, gen_jitter, opt_disc, config.LEARNING_RATE,
        )
        utils.load_checkpoint(
            config.CHECKPOINT_GEN_UNJITTER_LOAD, gen_unjitter, opt_disc, config.LEARNING_RATE,
        )

    # Define training dataset and loader
    train_dataset = JitteredDataset(1000, True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    # Define training dataset and loader
    val_dataset = JitteredDataset(1000, True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    gen_scaler = torch.cuda.amp.grad_scaler.GradScaler()
    disc_scaler = torch.cuda.amp.grad_scaler.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_function(
            disc_jitter, disc_unjitter, gen_jitter, gen_unjitter, train_loader,
            val_loader, opt_disc, opt_gen, L1, MSE, disc_scaler, gen_scaler, epoch, 
        )

        if config.SAVE_MODEL and epoch % 5 == 0:
            utils.save_checkpoint(gen_unjitter, opt_gen, filename=config.CHECKPOINT_GEN_UNJITTER_SAVE)
            utils.save_checkpoint(gen_jitter, opt_gen, filename=config.CHECKPOINT_GEN_JITTER_SAVE)
            utils.save_checkpoint(disc_unjitter, opt_disc, filename=config.CHECKPOINT_DISC_UNJITTER_SAVE)
            utils.save_checkpoint(disc_jitter, opt_disc, filename=config.CHECKPOINT_DISC_JITTER_SAVE)


if __name__ == "__main__":
    main()
