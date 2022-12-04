import argparse,os,sys




def parse_args():
    desc = "PyTorch implementation for XVFI"
    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument('--x_train_data_path', type=str, default="", help='path of X-Train dataset')
    parser.add_argument('--toptim', action='store_true', default=False, help='continue the training')

    #parser.add_argument('--gpu', type=int, default=0, help='path of X-Train dataset')
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Normal Training of 200 epochs
    command = "python main.py --phase 'train' --exp_num 2 --gpu 0 --papermodel --x_train_data_path " + args.x_train_data_path
    os.system(command)

    # Optional Toptimization  | it removes the latest checkpoint of the method of exp 2. If you want to do several training runs, change the exp_num to a new unique number.
    if args.toptim:
        #os.remove("checkpoint_dir\\fLDRnet_X4K1000FPS_exp2\\fLDRnet_X4K1000FPS_exp2_latest.pt")
        os.remove(os.path.join("checkpoint_dir","fLDRnet_exp2","fLDRnet_exp2_latest.pt"))
        command = "python main.py --phase 'train' --exp_num 2 --gpu 0 --papermodel --x_train_data_path " + args.x_train_data_path + " --epochs 220 --TOptimization --sminterpWT --init_lr 0.001"
        os.system(command)

if __name__ == '__main__':
    main()