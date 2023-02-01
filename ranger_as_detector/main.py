import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from functions import extract_ranger_bounds, load_bounds, run_with_range_supervision


def main():

    # Get input example image ------------------------------------------------------------------------------------------------------------------
    import urllib
    url, filename = ("https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/TCGA_CS_4944.png", "TCGA_CS_4944.png")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)

    input_image = Image.open(filename)
    m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=m, std=s),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)


    # Get model ------------------------------------------------------------------------------------------------------------------
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=1, init_features=32, pretrained=True)
    model.eval()


    # Extract and save bounds ------------------------------------------------------------------------------------------------------------------
    ranger_file_name = 'ranger_as_detector/unet_bounds_test.txt'
    dataset_for_bound_extraction = [input_batch]
    extract_ranger_bounds(dataset_for_bound_extraction, model, ranger_file_name) #can be done only once, include later real dataloader


    # Run inference with range supervision --------------------------------------------------------------------------------------------------------
    bnds = load_bounds(ranger_file_name)
    # bnds[0][1] = 0.1 #Simple test case

    output, act_list = run_with_range_supervision(model, input_batch, bnds, mitigation=False) #include later real dataloader
    print('prediction result:', output)
    print('Out-of-bound values encountered:', any(act_list))



if __name__ == "__main__":
    main()