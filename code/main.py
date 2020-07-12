import argparse
import os
from utils import select_GPUs

os.environ['CUDA_VISIBLE_DEVICES'] = str(select_GPUs(1, .1, 0.6)[0])
print("GPU:%s" % os.environ['CUDA_VISIBLE_DEVICES'])

from model_VD import ModelBaseline_VD, ModelBaseline_VD_Mixup
from model_PACS import ModelBaseline_PACS
import sys
import torch

sys.setrecursionlimit(1000000)
import warnings

warnings.filterwarnings("ignore")
torch.set_num_threads(4)


def main():
    train_arg_parser = argparse.ArgumentParser()
    train_arg_parser.add_argument("--dataset", type=str, default='VD',
                                  help='VD')
    train_arg_parser.add_argument("--method", type=str, default='baseline',
                                  help='baseline')
    args = train_arg_parser.parse_args()
    #
    # check_list : save_path/pre_trained model name/get feature/part forward/mixup alpha/meta flag/reconstruct flag/freeze
    #
    if args.dataset == 'VD':
        train_arg_parser.add_argument("--meta", type=bool, default=True,
                                      help='Whether to use meta learning ')
        train_arg_parser.add_argument("--meta_beta", type=float, default=1.,
                                      help='Whether to use meta learning ')
        train_arg_parser.add_argument("--reconstruct", type=bool, default=True,
                                      help='Whether to reconstruct ')
        train_arg_parser.add_argument("--reconstruct_tradeoff", type=float, default=0.01,
                                      help='reconstruct loss tradeoff ')
        train_arg_parser.add_argument("--adv_tradeoff", type=float, default=0.1,
                                      help='adv loss tradeoff ')
        train_arg_parser.add_argument("--mixup", type=bool, default=True,
                                      help='Whether to use mixup')
        train_arg_parser.add_argument("--mixup_alpha", type=float, default=8,  # 1.0,
                                      help='Whether to use mixup')
        train_arg_parser.add_argument("--mixup_method", type=str, default='multi',  # 'multi',
                                      help='Whether to use mixup')
        train_arg_parser.add_argument("--batch_size", type=int, default=64,
                                      help="batch size for training, default is 64")
        train_arg_parser.add_argument("--batch_size_metatest", type=int, default=32,
                                      help="batch size for meta testing, default is 32")
        train_arg_parser.add_argument("--num_classes", type=int, default=10,
                                      help="number of classes")
        train_arg_parser.add_argument("--iteration_size", type=int, default=40000,  # used to be 20000
                                      help="iteration of training domains")
        train_arg_parser.add_argument("--lr", type=float, default=5e-4,
                                      help='learning rate of the model')
        train_arg_parser.add_argument("--beta", type=float, default=100,
                                      help='learning rate of the dg function')
        train_arg_parser.add_argument("--heldout_p", type=float, default=1,
                                      help='learning rate of the heldout function')
        train_arg_parser.add_argument("--omega", type=float, default=1e-4,
                                      help='learning rate of the omega function')
        train_arg_parser.add_argument("--meta_iteration_size", type=int, default=1,
                                      help='iteration of test domains')
        filename = ['mixup_best_meta_featrue', 'mixup_shallow5_featrue_e2e_decoder', 'mixup_shallow5_featrue_e2e',
                    'mixup_shallow14_featrue', 'mixup_last_feature', 'mixup_before_pooling',
                    'meta_deocder','meta_decoder_adv_new','decoder_residual',
                    'final','final_no_res','final_with_res',
                    'final_hope']
        train_arg_parser.add_argument("--logs", type=str, default='logs/CVPR/%s/' % filename[12],
                                      help='logs folder to write log')
        train_arg_parser.add_argument("--model_path", type=str, default='model_output/CVPR/%s/' % filename[12],
                                      help='folder for saving model')
        train_arg_parser.add_argument("--debug", type=bool, default=True,
                                      help='whether for debug mode or not')
        train_arg_parser.add_argument("--count_test", type=int, default=1,
                                      help='the amount of episode for testing our method')
        train_arg_parser.add_argument("--if_train", type=bool, default=True,
                                      help='if we need to train to get the model')
        train_arg_parser.add_argument("--if_test", type=bool, default=True,
                                      help='if we want to test on the target data')
        args = train_arg_parser.parse_args()

        for i in range(args.count_test):
            if args.mixup:
                model_obj = ModelBaseline_VD_Mixup(flags=args)
            else:
                model_obj = ModelBaseline_VD(flags=args)
            if args.if_train == True:
                model_obj.train(flags=args)
                torch.cuda.empty_cache()
            if args.if_test == True:
                model_obj.heldout_test(flags=args)
                model_obj.heldout_test_knn_cos(flags=args)


if __name__ == "__main__":
    main()
