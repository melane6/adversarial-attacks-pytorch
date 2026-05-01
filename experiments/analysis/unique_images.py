import json

import pandas as pd

heatmap_key="responsibility"
mask_key=""
MODELS_conv=["convnext-tiny", "convnext-small", "convnext-base", "convnext-large"]
MODELS_resnet=["resnet18", "resnet50", "resnet101", "resnet152"]
MODELS_vgg=["vgg11", "vgg13", "vgg16", "vgg19"]
MODELS=MODELS_conv+MODELS_resnet+MODELS_vgg
for model in MODELS:
    images = []
    for mask in ["", "-mask", "-mask-2", "-mask-3", "-mask-4", "-mask-5"]:
        output_dir=f"/mnt/data/Documents/adversarial-attacks-pytorch/{model}/{model}-one-pixel{mask}"
        model_e=model.replace("-","_")
        json_path = f"{output_dir}/results_onepixel_{model_e}.json"
        with open(json_path, 'r') as f:
            results = json.load(f)
        statistics = results['statistics']
        metadata = results['metadata']
        samples = pd.json_normalize(results['samples'])
        # filter for successful attacks
        successful_attacks = samples[samples['attack_success'] == True]
        images.append(successful_attacks['image_name'].tolist())

    # Summary statistics
    print(f"Model: {model}")
    for i, image in enumerate(images):
        print(f"Mask: {i}")
        print(f"Number of successful attacks: {len(image)}")
        unique_images = set(images[0] + image)
        print(f"Number of unique images: {len(unique_images)}")

    total_uniques_images = set(images[1] + images[2] + images[3] + images[4] + images[5])
    print(f"Unique images for MSPS: {len(total_uniques_images)}")
    print("=="*50)