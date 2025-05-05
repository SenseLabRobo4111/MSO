import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
# from losses.perceptual_loss import PerceptualLoss
from losses.adversarial import make_discrim_loss
from losses.resnet_pl import ResNetPL
# from losses.feature_matching import feature_matching_loss
from losses.feature_matching import masked_l1_loss
from tqdm import tqdm
import matplotlib.pyplot as plt


def train_model(gen, critic, dataloaders, num_epochs=25, device='cpu', save_dir=None):
    torch.backends.cudnn.benchmark = True  # 加速训练

    # perceptual_criterion = PerceptualLoss().eval().to(device)
    BCE_criterion = nn.BCEWithLogitsLoss()
    ResNetPL_criterion = ResNetPL(weight=30, weights_path="/home/senselabrobo/iros/HouseExpo/explore_model/losses/models/").to(device)
    Adversarial_criterion = make_discrim_loss('r1', gp_coef=0.01, weight=10, mask_as_fake_target=True, allow_scale_mask=True)

    gen_optimizer = optim.Adam(gen.parameters(), lr=0.0002)
    critic_optimizer = optim.Adam(critic.parameters(), lr=0.0002)

    gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, step_size=100, gamma=0.1)
    critic_scheduler = optim.lr_scheduler.StepLR(critic_optimizer, step_size=100, gamma=0.1)

    gen.to(device)  # 将模型转移到设备（GPU 或 CPU）
    critic.to(device)

    gen.train()
    critic.train()

    for epoch in range(num_epochs):
        running_critic_loss = 0.0
        running_gen_loss = 0.0
            
        for features, labels, masks, _ in tqdm(dataloaders['train']):

            # 将数据移到相同的设备
            features = features.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            critic_optimizer.zero_grad()

            fake_images1 = gen(features).detach()

            true_img1 = labels
            Adversarial_criterion.pre_discriminator_step(real_batch=true_img1, fake_batch=fake_images1,
                                                 generator=gen, discriminator=critic)
            critic_real1 = critic(true_img1, masks)
            critic_fake1 = critic(fake_images1, masks)
            
            adv_discr_loss, adv_metrics = Adversarial_criterion.discriminator_loss(real_batch=true_img1,
                                                                           fake_batch=fake_images1,
                                                                           discr_real_pred=critic_real1,
                                                                           discr_fake_pred=critic_fake1,
                                                                           mask=masks)

            total_critic_loss = adv_discr_loss

            # 反向传播
            total_critic_loss.backward()
            critic_optimizer.step()

            gen_optimizer.zero_grad()

            # # 冻结Critic参数以节省计算资源
            # for param in critic.parameters():
            #     param.requires_grad = False

            fake_images2 = gen(features)

            true_img2 = labels
            Adversarial_criterion.pre_generator_step(real_batch=true_img2, fake_batch=fake_images2,
                                             generator=gen, discriminator=critic)
            critic_real2 = critic(true_img2, masks)
            critic_fake2 = critic(fake_images2, masks)
            adv_gen_loss, adv_metrics = Adversarial_criterion.generator_loss(real_batch=true_img2,
                                                                     fake_batch=fake_images2,
                                                                     discr_real_pred=critic_real2,
                                                                     discr_fake_pred=critic_fake2,
                                                                     mask=masks)
            total_gen_loss = adv_gen_loss

            resnet_pl_value = ResNetPL_criterion(fake_images2, true_img2)
            total_gen_loss = total_gen_loss + resnet_pl_value
            
            l1_loss = masked_l1_loss(fake_images2, true_img2, masks, 10, 0)
            total_gen_loss = total_gen_loss + l1_loss

            # 混合精度反向传播
            total_gen_loss.backward()
            gen_optimizer.step()

            # # 解冻Critic参数
            # for param in critic.parameters():
            #     param.requires_grad = True
            
            running_critic_loss += total_critic_loss.item()
            running_gen_loss += total_gen_loss.item()
        
        epoch_critic_loss = running_critic_loss / len(dataloaders['train'])
        epoch_gen_loss = running_gen_loss / len(dataloaders['train'])

        print(f"Epoch {epoch+1}/{num_epochs}, Critic Loss: {epoch_critic_loss:.4f}, Generator Loss: {epoch_gen_loss:.4f}")
        
        critic_scheduler.step()
        gen_scheduler.step()

        # 保存模型
        if save_dir is not None and ((epoch+1) % 5 == 0 or epoch+1 < 20):
            torch.save(gen.state_dict(), save_dir + f"epoch_{epoch+1}.pth")
            feature_np = features.cpu().detach().numpy().transpose(0, 2, 3, 1) * 255
            label_np = labels.cpu().detach().numpy().transpose(0, 2, 3, 1) * 255
            output_np = fake_images1.cpu().detach().numpy().transpose(0, 2, 3, 1) * 255

            plt.figure(figsize=(18, 6))

            # 特征标签
            plt.subplot(1, 3, 1)
            plt.imshow(feature_np[0])  # 使用 imshow 显示图像
            plt.title('Feature Labels')
            plt.draw()

            # 真实标签
            plt.subplot(1, 3, 2)
            plt.imshow(label_np[0], cmap='gray', vmin=0, vmax=255)  # 使用 imshow 显示图像
            plt.title('True Labels')
            plt.draw()

            # 预测标签
            plt.subplot(1, 3, 3)
            plt.imshow(output_np[0], cmap='gray', vmin=0, vmax=255)  # 使用 imshow 显示图像
            plt.title('Predicted Labels')
            plt.draw()

            plt.savefig(save_dir + f"epoch_{epoch+1}.png")
            plt.close()
        
    print("Training complete!")
