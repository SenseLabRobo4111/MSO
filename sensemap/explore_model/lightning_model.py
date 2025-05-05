import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .dataset import MapReconstructionDataset
from .SenseMapNet import SenseMapNet
from .critic_model import CriticModel
from .losses.adversarial import make_discrim_loss
from .losses.resnet_pl import ResNetPL
from .losses.feature_matching import masked_l1_loss
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import sys

def set_requires_grad(module, value):
    for param in module.parameters():
        param.requires_grad = value

class SenseMapNetDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir="dataset3", batch_size=16, num_workers=4):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = MapReconstructionDataset(os.path.join(self.dataset_dir, "train"))
        self.val_dataset = MapReconstructionDataset(os.path.join(self.dataset_dir, "test"))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

class SenseMapNetModule(pl.LightningModule):
    def __init__(self, gen, critic, save_dir="explore_model/model_logs/ffc/"):
        super().__init__()
        self.save_hyperparameters(ignore=['gen', 'critic'])
        self.gen = gen
        self.critic = critic
        self.save_dir = save_dir
        self.automatic_optimization = False  # 手动优化

        # 初始化损失函数
        self.BCE_criterion = nn.BCEWithLogitsLoss()
        self.ResNetPL_criterion = ResNetPL(
            weight=30,
            weights_path="/home/senselabrobo/iros/HouseExpo/explore_model/losses/models/"
        ).eval()
        self.Adversarial_criterion = make_discrim_loss(
            "r1",
            gp_coef=0.01,
            weight=10,
            mask_as_fake_target=True,
            allow_scale_mask=True
        )

    def forward(self, x):
        return self.gen(x)

    def training_step(self, batch, batch_idx):
        # 获取优化器
        gen_opt, critic_opt = self.optimizers()
        features, labels, masks, _ = batch

        # 训练判别器
        # ---------------------
        critic_opt.zero_grad()
        with torch.no_grad():
            fake_images = self.gen(features)
        
        # 计算判别器损失
        self.Adversarial_criterion.pre_discriminator_step(
            real_batch=labels,
            fake_batch=fake_images,
            generator=self.gen,
            discriminator=self.critic
        )
        
        real_pred = self.critic(labels, masks)
        fake_pred = self.critic(fake_images.detach(), masks)
        d_loss, _ = self.Adversarial_criterion.discriminator_loss(
            real_batch=labels,
            fake_batch=fake_images,
            discr_real_pred=real_pred,
            discr_fake_pred=fake_pred,
            mask=masks
        )
        
        self.manual_backward(d_loss)
        critic_opt.step()

        # 训练生成器
        # ---------------------
        gen_opt.zero_grad()
        fake_images = self.gen(features)
        
        # 计算生成器损失
        self.Adversarial_criterion.pre_generator_step(
            real_batch=labels,
            fake_batch=fake_images,
            generator=self.gen,
            discriminator=self.critic
        )
        
        real_pred = self.critic(labels, masks)
        fake_pred = self.critic(fake_images, masks)
        g_adv_loss, _ = self.Adversarial_criterion.generator_loss(
            real_batch=labels,
            fake_batch=fake_images,
            discr_real_pred=real_pred,
            discr_fake_pred=fake_pred,
            mask=masks
        )
        
        # 其他损失项
        resnet_pl = self.ResNetPL_criterion(fake_images, labels)
        l1_loss = masked_l1_loss(fake_images, labels, masks, 10, 0)
        total_g_loss = g_adv_loss + resnet_pl + l1_loss

        self.manual_backward(total_g_loss)
        gen_opt.step()

        # 记录损失
        self.log_dict({
            "d_loss": d_loss,
            "g_adv_loss": g_adv_loss,
            "resnet_pl": resnet_pl,
            "l1_loss": l1_loss,
            "g_total_loss": total_g_loss
        }, prog_bar=True)

    def configure_optimizers(self):
        gen_opt = optim.Adam(self.gen.parameters(), lr=0.0002, weight_decay=0.001)
        critic_opt = optim.Adam(self.critic.parameters(), lr=0.0002, weight_decay=0.001)
        
        gen_sch = {"scheduler": optim.lr_scheduler.StepLR(gen_opt, step_size=50, gamma=0.4), "interval": "step"}
        critic_sch = {"scheduler": optim.lr_scheduler.StepLR(critic_opt, step_size=50, gamma=0.4), "interval": "step"}
        
        return [gen_opt, critic_opt], [gen_sch, critic_sch]
        # return gen_opt, critic_opt

class ImageCallback(pl.Callback):
    def __init__(self, save_dir, every_n_epochs=5):
        super().__init__()
        self.save_dir = save_dir
        self.every_n_epochs = every_n_epochs
        os.makedirs(save_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epochs == 0 or (epoch + 1) < 20:
            # 获取示例数据
            sample = next(iter(trainer.datamodule.train_dataloader()))
            features, labels, masks, _ = sample
            features = features.to(pl_module.device)
            
            with torch.no_grad():
                outputs = pl_module.gen(features)
            
            # 转换并保存图像
            self._save_images(features[0], labels[0], outputs[0], epoch+1)

    def _save_images(self, feature, label, output, epoch):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        ax1.imshow(feature.cpu().permute(1, 2, 0).numpy())
        ax1.set_title("Input Features")
        ax1.axis("off")
        
        ax2.imshow(label.cpu().permute(1, 2, 0).numpy(), cmap="gray")
        ax2.set_title("Ground Truth")
        ax2.axis("off")
        
        ax3.imshow(output.cpu().permute(1, 2, 0).numpy(), cmap="gray")
        ax3.set_title("Generated Output")
        ax3.axis("off")
        
        plt.savefig(os.path.join(self.save_dir, f"epoch_{epoch}.png"))
        plt.close()

def start_train(gen, save_dir):
    # 创建数据模块和模型
    dm = SenseMapNetDataModule()
    model = SenseMapNetModule(gen, CriticModel(), save_dir=save_dir)

    # 配置回调函数
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="model-{epoch:03d}",
        save_top_k=-1,
        every_n_epochs=5
    )
    image_callback = ImageCallback(save_dir)
    
    # 训练器配置
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=300,
        callbacks=[checkpoint_callback, image_callback],
        enable_progress_bar=True,
        logger=True,
        enable_model_summary=True,
        strategy="ddp_find_unused_parameters_true",
        # precision='16-mixed'
        #profiler="simple"
        log_every_n_steps=900,
    )

    # 开始训练
    trainer.fit(model, dm)

if __name__ == "__main__":
    # 初始化生成器
    gen = SenseMapNet(dim=int(sys.argv[1]), embed_dim=int(sys.argv[2]), num_heads=int(sys.argv[3]), num_layers=int(sys.argv[3]))
    start_train(gen, save_dir=f"explore_model/model_logs/ffc-{sys.argv[4]}/")