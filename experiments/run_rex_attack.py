import os
import glob
import json
import argparse
import subprocess
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_folder", type=str, required=True)
    parser.add_argument("--model", type=str, default="resnet152")
    parser.add_argument("--script", type=str, default="")
    args = parser.parse_args()

    attack_info_files = glob.glob(os.path.join(args.attack_folder, "*.json"))
    if not attack_info_files:
        print(f"No JSON files found in {args.attack_folder}. Skipping.")
        return

    attack_info = attack_info_files[0]
    with open(attack_info, 'r') as f:
        results = json.load(f)

    samples = pd.json_normalize(results.get('samples', []))

    OUTPUT_DIR = os.path.join(args.attack_folder, f"{args.model}_exp")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    DUMP_PATH = os.path.join(OUTPUT_DIR, f"{args.model}_attacked_rex.csv")
    # Run on all the Successful attacks
    for idx, sample in samples.iterrows():
        if not sample.get('attack_success', False):
            continue

        image_name = str(sample['image_name']).replace('.JPEG', '')
        class_name = sample['class_name']
        pred_clean = sample['pred_clean']
        pred_adv = sample['pred_adv']
        image_path = sample['image_path']

        output_path = os.path.join(OUTPUT_DIR, f"{image_name}_rex.jpeg")
        if os.path.exists(output_path):
            print(f"ReX output already exists for {image_name}, skipping...")
            continue
        print(f"[{args.model}] Image: {image_name} | Clean: {pred_clean} -> Adv: {pred_adv}")
        try:
            subprocess.run([
                "ReX", image_path,
                "--script", args.script,
                "-v",
                "--dump", DUMP_PATH,
                "--config", "/mnt/data/Documents/ReX/rex.toml",
                "--custom_target", str(pred_adv),
                "--no_extract",
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"ReX failed on {image_name}: {e}")


if __name__ == "__main__":
    main()