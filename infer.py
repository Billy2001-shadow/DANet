import cv2
import argparse
import numpy as np
import os
import torch
from models.net import DANet
from torchvision import transforms
from PIL import Image

def _get_test_opt():
    parser = argparse.ArgumentParser(description = 'Evaluate performance of DANet')
    parser.add_argument('--backbone', default='EfficientNet', help='select a network as backbone')
    parser.add_argument('--loadckpt', default='/home/chenwu/DANet/weights/NYUD.pt',required=False, help="the path of the loaded model")
    parser.add_argument('--threshold', type=float, default=1.0, help="threshold of the pixels on edges")
    parser.add_argument('--pretrained_dir', type=str,default='./pretrained', required=False, help="the path of pretrained models")
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)
    parser.add_argument('--height', type=float, help='feature height', default=8)
    parser.add_argument('--width', type=float, help='feature width', default=10)
    # parse arguments
    return parser.parse_args()


def preprocess_torch(image_path):
    """
    对单张图像进行预处理，与测试过程中一致。
    
    Args:
        image_path (str): 图像的路径。
        __imagenet_stats (dict): 图像的均值和标准差。
    
    Returns:
        torch.Tensor: 预处理后的图像张量。
    """
    # 定义统计量
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    
    preprocess = transforms.Compose([
        transforms.Resize(240),  # 调整大小
        transforms.CenterCrop((228, 304)),  # 中心裁剪到指定尺寸
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(__imagenet_stats['mean'], __imagenet_stats['std'])  # 标准化
    ])
    
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    
    # 进行预处理
    image_tensor = preprocess(image)
    
    image_tensor = image_tensor.unsqueeze(0)  # 添加批次维度

    print(image_tensor.shape)

    return image_tensor

# 基于torch的数据后处理: 
def postprocess_torch(output: torch.Tensor) -> np.ndarray:
    output = torch.nn.functional.interpolate(output[-1], size=(228, 304), mode='bilinear', align_corners=True)
    depth_map = output.squeeze().cpu().detach().numpy()
    return depth_map


def visualization(depth: np.ndarray, orig_h: int, orig_w: int, orig_imgpath: str):

    depth = cv2.resize(depth, (orig_w, orig_h))
    print(depth.shape)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0  # Normalize?
    depth = depth.astype(np.uint8)                                       # Convert to uint8  
    depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

    # Visualization
    
    margin_width = 50
    caption_height = 60
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    split_region = np.ones((orig_h, margin_width, 3), dtype=np.uint8) * 255
    combined_results = cv2.hconcat([cv2.imread(orig_imgpath), split_region, depth_color])

    caption_space = (
        np.ones((caption_height, combined_results.shape[1], 3), dtype=np.uint8)
        * 255
    )
    captions = ["Raw image", "DANet"]
    segment_width = orig_w + margin_width
    for i, caption in enumerate(captions):
        # Calculate text size
        text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]

        # Calculate x-coordinate to center the text
        text_x = int((segment_width * i) + (orig_w - text_size[0]) / 2)

        # Add text caption
        cv2.putText(
            caption_space,
            caption,
            (text_x, 40),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
        )

    final_result = cv2.vconcat([caption_space, combined_results])

    cv2.imwrite("/home/chenwu/DANet/test_results/depth_color.png", final_result)


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES']='1'

    # 指定输入图像路径和输出深度图路径
    images_path = '/home/chenwu/DANet/test_images/250.png'
    output_path = '/home/chenwu/DANet/test_results'

    # 1.获取参数
    args = _get_test_opt()
    # 2.加载数据(并通过数据增强)
    image_tensor = preprocess_torch(images_path)
    # 3.加载模型
    model = DANet(args)
    model = torch.nn.DataParallel(model)
    model.cuda()
    model.eval()
    print("loading model {}".format(args.loadckpt))
    state_dict=torch.load(args.loadckpt)["model"]
    model.load_state_dict(state_dict)
    

    with torch.no_grad():
        bin,output= model(image_tensor)
        depth_map = postprocess_torch(output)
        visualization(depth_map, 480, 640, images_path)
        

