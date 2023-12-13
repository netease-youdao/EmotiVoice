import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--special_tokens',
                        type=str,
                        help='Path to special_token.txt')
    return parser.parse_args()

def main(args):
    with open(args.special_tokens, "w") as f:
        for line in {"sp0", "sp1", "sp2", "sp3", "sp4","engsp1", "engsp2", "engsp3", "engsp4", "<sos/eos>", "cn_eng_sp", "eng_cn_sp", "." , "?", "LAUGH"}:
            f.write(f"{line}\n")

if __name__ == '__main__':
    main(get_args())
