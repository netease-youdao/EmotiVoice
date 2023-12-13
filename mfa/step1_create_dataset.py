
from tqdm import tqdm 
import jsonlines
import re
import argparse
import os

def main(args):
    ROOT_DIR=os.path.abspath(args.data_dir)
    TEXT_DIR=f"{ROOT_DIR}/text"
    MFA_DIR=f"{ROOT_DIR}/mfa"

    os.makedirs(MFA_DIR, exist_ok=True)
    


    with jsonlines.open(f"{TEXT_DIR}/datalist.jsonl") as f1, \
        open(f"{MFA_DIR}/text_sp1-sp4", "w") as f2, \
        open(f"{MFA_DIR}/wav.scp", "w") as f3:

        data=list({f'{sample["key"]}_{sample["speaker"]}':sample for sample in list(f1)}.values())
        
        for sample in tqdm(data):
            text=[]
            for ph in sample["text"]:
                if ph[0] == '[':
                    ph = ph[1:-1]
                elif ph == "cn_eng_sp":
                    ph = "cnengsp"
                elif ph == "eng_cn_sp":
                    ph = "engcnsp"
                text.append(ph)
            f2.write("{}|{} {}\n".format(re.sub(r" +", "", sample["speaker"]), sample["key"], " ".join(text)))
            f3.write("{} {}\n".format(sample["key"], sample["wav_path"]))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True)
    args = p.parse_args()

    main(args)
