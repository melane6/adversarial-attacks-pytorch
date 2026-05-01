import pandas as pd

heatmap_key="responsibility"
mask_key=""
MODELS_conv=["convnext-tiny", "convnext-small", "convnext-base", "convnext-large"]
MODELS_resnet=["resnet18", "resnet50", "resnet101", "resnet152"]
MODELS_vgg=["vgg11", "vgg13", "vgg16", "vgg19"]
MODELS=MODELS_conv+MODELS_resnet+MODELS_vgg

for model in MODELS:
    print(model)
    output_dir=f"/mnt/data/Documents/adversarial-attacks-pytorch/{model}/{model}-one-pixel/analysis_attacked/"
    df_path = f"{output_dir}analysis_{heatmap_key}{mask_key}.csv"
    df = pd.read_csv(df_path)
    print(df)
    df['within_mask'] = df['within_mask'].map(lambda x: True if 'True' in x else False)
    df['within_heatmap'] = df['within_heatmap'].map(lambda x: True if 'True' in x else False)
    count_in_mask = df.value_counts('within_mask')[True] if True in df['within_mask'].value_counts() else 0
    count_in_heatmap = df.value_counts('within_heatmap')[True] if True in df['within_heatmap'].value_counts() else 0
    avg_resp_in_pert = df['heatmap_at_perturbation'].mean()
    std_resp_in_pert = df['heatmap_at_perturbation'].std()
    print(f"Average responsibility in perturbations: {avg_resp_in_pert:.4f} & {std_resp_in_pert:.4f}")
    print(f"Perturbations within mask: {count_in_mask}/{len(df)} ({count_in_mask / len(df) * 100:.2f}%)")
    print(f"Perturbations within heatmap: {count_in_heatmap}/{len(df)} ({count_in_heatmap / len(df) * 100:.2f}%)")