

import jsonlines
import argparse
import os


def main(args):
    ROOT_DIR=os.path.abspath(args.data_dir)
    TEXT_DIR=f"{ROOT_DIR}/text"
    MFA_DIR=f"{ROOT_DIR}/mfa"
    TRAIN_DIR=f"{ROOT_DIR}/train"
    VALID_DIR=f"{ROOT_DIR}/valid"

    with jsonlines.open(f"{MFA_DIR}/datalist.jsonl") as f:
        data = list(f)
    
    with jsonlines.open(f"{TEXT_DIR}/datalist.jsonl") as f:
        data_ref = {sample["key"]:sample for sample in list(f)}

    new_data = []
    with jsonlines.open(f"{TEXT_DIR}/datalist_mfa.jsonl", "w") as f:
        for sample in data:
            if "duration" in sample:
                del sample["duration"]
            

            
            # if "emotion" not in sample:
            #     sample["emotion"]="default"

            
            for i, ph in enumerate(sample["text"]):
                if ph.isupper():
                    sample["text"][i] = "[" + ph + "]"
                    
                if ph =="cnengsp":
                    sample["text"][i] = "cn_eng_sp"
                if ph =="engcnsp":
                    sample["text"][i] = "eng_cn_sp"
            
            sample_ref = data_ref[sample["key"]]
            
            sample["original_text"]=sample_ref["original_text"]
            sample["prompt"] = sample_ref["prompt"]
            new_data.append(sample)
            f.write(sample)
    
    with jsonlines.open(f"{TRAIN_DIR}/datalist_mfa.jsonl", "w") as f:
        for sample in new_data[:-3]:
            f.write(sample)
            
    with jsonlines.open(f"{VALID_DIR}/datalist_mfa.jsonl", "w") as f:
        for sample in data[-3:]:
            f.write(sample)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True)
    args = p.parse_args()

    main(args)