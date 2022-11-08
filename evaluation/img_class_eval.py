import sys, os
import argparse
sys.path.append(os.getcwd())
from alficore.evaluation.img_class_eval import img_class_eval as imgclass_eval

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', type=str, help='path to result files')
    opt = parser.parse_args()
    return opt

def img_class_eval(argv):
    opt = parse_opt()
    folder = opt.res_dir
    imgclass_eval(exp_folder_path=folder)

if __name__ == "__main__":
    img_class_eval(sys.argv)
