import torch
import config
import utils
from datasetFile import HorseZebraDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm 
from discriminator import Discriminator
from generator import Generator
from torchvision.utils import save_image

def train_function(disc_H, disc_Z, gen_H, gen_Z, loader, opt_disc, opt_gen,
                       L1, MSE, disc_scaler, gen_scaler, epoch):
    
    loop = tqdm(loader, leave=True)
     
    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # Train discriminator H and Z
        with torch.cuda.amp.autocast():
            # Find disc loss for horses
            # Generate fake horse image
            fake_horse = gen_H(zebra)
            # Get disc score for real and fake horse image
            disc_horse_real = disc_H(horse)
            # Use detach and this will be used to train generator aswell
            disc_horse_fake = disc_H(fake_horse.detach())
            
            disc_horse_real_loss = MSE(disc_horse_real, torch.ones_like(disc_horse_real))
            disc_horse_fake_loss = MSE(disc_horse_fake, torch.zeros_like(disc_horse_fake))

            disc_horse_loss = disc_horse_real_loss + disc_horse_fake_loss

            # Find disc loss for zebras
            # Generate fake zebras image
            fake_zebra = gen_H(horse)
            # Get disc score for real and fake horse image
            disc_zebra_real = disc_H(zebra)
            # Use detach and this will be used to train generator aswell
            disc_zebra_fake = disc_H(fake_zebra.detach())
            
            disc_zebra_real_loss = MSE(disc_zebra_real, torch.ones_like(disc_zebra_real))
            disc_zebra_fake_loss = MSE(disc_zebra_fake, torch.zeros_like(disc_zebra_fake))

            disc_zebra_loss = disc_zebra_real_loss + disc_zebra_fake_loss

            # Total discriminator loss
            disc_loss = (disc_horse_loss + disc_zebra_loss)/2

        # Compute backpropogation for generators 
        opt_disc.zero_grad()
        disc_scaler.scale(disc_loss).backward()
        disc_scaler.step(opt_disc)
        disc_scaler.update()

        # Train generator H and Z
        with torch.cuda.amp.autocast():
            # Adversarial loss for both generators
            disc_horse_fake = disc_H(fake_horse)
            disc_zebra_fake = disc_Z(fake_zebra)
            
            gen_zebra_loss = MSE(disc_zebra_fake, torch.ones_like(disc_zebra_fake))
            gen_horse_loss = MSE(disc_horse_fake, torch.ones_like(disc_horse_fake))

            # Cycle loss
            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)

            cycle_zebra_loss = L1(zebra, cycle_zebra)
            cycle_horse_loss = L1(horse, cycle_horse)
            
            # Identity loss
            identity_zebra = gen_Z(zebra)
            identity_horse = gen_H(horse)

            identity_zebra_loss = L1(zebra, identity_zebra)
            identity_horse_loss = L1(zebra, identity_horse)

            # Total generator loss
            gen_loss = (
                gen_zebra_loss + 
                gen_horse_loss +
                cycle_zebra_loss * config.LAMBDA_CYCLE +
                cycle_horse_loss * config.LAMBDA_CYCLE + 
                identity_zebra_loss * config.LAMBDA_IDENTITY + 
                identity_horse_loss * config.LAMBDA_IDENTITY
            )
        # Compute backpropogation for generators
        opt_gen.zero_grad()
        gen_scaler.scale(gen_loss).backward()
        gen_scaler.step(opt_gen)
        gen_scaler.update()

    if epoch % 5 ==0:
        save_image(fake_zebra, f"../evaluation/fake_zebra_{epoch}.png")
        save_image(fake_horse, f"../evaluation/fake_horse_{epoch}.png")


def main():
    disc_H = Discriminator(inChannels=config.CHANNELS_IMG).to(config.DEVICE) # for horse
    disc_Z = Discriminator(inChannels=config.CHANNELS_IMG).to(config.DEVICE) # for zebra
    
    gen_H = Generator(
        imageChannels=config.CHANNELS_IMG,
        numResiduals=config.NUM_RESIDUALS,
    ).to(config.DEVICE)
    gen_Z = Generator(
        imageChannels=config.CHANNELS_IMG,
        numResiduals=config.NUM_RESIDUALS,
    ).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr = config.LEARNING_RATE,
        betas = config.OPTIMISER_WEIGHTS
    )
    opt_gen = optim.Adam(
        list(gen_H.parameters()) + list(gen_Z.parameters()),
        lr = config.LEARNING_RATE,
        betas = config.OPTIMISER_WEIGHTS
    )

    L1 = nn.L1Loss()
    MSE = nn.MSELoss()

    if config.LOAD_MODEL:
        utils.load_checkpoint(
            config.CHECKPOINT_DISC_H_LOAD, disc_H, opt_disc, config.LEARNING_RATE,
        )
        utils.load_checkpoint(
            config.CHECKPOINT_DISC_Z_LOAD, disc_Z, opt_disc, config.LEARNING_RATE,
        )

        utils.load_checkpoint(
            config.CHECKPOINT_GEN_H_LOAD, gen_H, opt_disc, config.LEARNING_RATE,
        )
        utils.load_checkpoint(
            config.CHECKPOINT_GEN_Z_LOAD, gen_Z, opt_disc, config.LEARNING_RATE,
        )

    dataset = HorseZebraDataset(
        root_horse=config.TRAIN_DIR_HORSE,
        root_zebra=config.TRAIN_DIR_ZEBRA,
        transform=config.transforms_concatinated,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    gen_scaler = torch.cuda.amp.grad_scaler.GradScaler()
    disc_scaler = torch.cuda.amp.grad_scaler.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_function(disc_H, disc_Z, gen_H, gen_Z, loader, opt_disc, opt_gen,
                       L1, MSE, disc_scaler, gen_scaler, epoch)

        if config.SAVE_MODEL and epoch % 5 == 0:
            utils.save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H_SAVE)
            utils.save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z_SAVE)
            utils.save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_DISC_H_SAVE)
            utils.save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_DISC_Z_SAVE)


if __name__ == "__main__":
    main()
