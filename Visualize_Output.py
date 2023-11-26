import matplotlib.pyplot as plt
from torchvision import transforms
from DataLoader import NYUDataset
import random
from Model import *
import torch



def visualize_depth_prediction(model, input_image, depth_map):
    model.eval()
    ip_img = input_image

    # input_image = input_image.clamp(0, 255).to(torch.uint8)
    # input_image = input_image.to(torch.uint8)

    input_image = input_image.unsqueeze(0).cpu().float()
    depth_map = depth_map.squeeze().cpu().float().numpy()

    with torch.no_grad():
        try:
            # print(input_image.shape)
            output_depth = model(input_image)
        except Exception as e:
            print(f"Error during forward pass: {e}")
            return

    output_depth = output_depth.squeeze().cpu().numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    # plt.imshow(input_image.squeeze(0).permute(1, 2, 0).numpy())
    plt.imshow(ip_img.numpy().transpose((1, 2, 0)))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(depth_map, cmap='jet')  # Use jet colormap for depth maps
    plt.title('Actual Depth Map')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(output_depth, cmap='jet')  # Use jet colormap for depth maps
    plt.title('Model Output Depth Map')
    plt.axis('off')

    plt.show()


def Visualize_Depth():
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize((256, 320)),
    ])
    val_dataset = NYUDataset(split='validation', transform=transform)
    random_val_indices = random.sample(range(len(val_dataset)), 10)
    random_val_samples = [val_dataset[i] for i in random_val_indices]
    val_input_images, val_depth_maps = zip(*random_val_samples)
    path = 'monodepth_model.pth'
    model = MonoDepthModel()
    model.load_state_dict(torch.load(path))
    model.eval()
    visualize_depth_prediction(model, val_input_images[0], val_depth_maps[0])


