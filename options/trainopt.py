import argparse

def _get_train_opt():
    parser = argparse.ArgumentParser(description = 'Monocular Depth Distribution Alignment with Low Computation(DANet)')
    parser.add_argument('--logging', default=False, help="log in Wandb")
    parser.add_argument('--backbone', type=str, default='EfficientNet', help='select a network as backbone')
    parser.add_argument('--checkpoint_dir', required=False,default='./results/EfficientNet/', help="the directory to save the checkpoints")
    parser.add_argument('--trainlist_path',default='./data/nyu2_train.csv', required=False, help='the path of trainlist')
    parser.add_argument('--testlist_path', default='./data/nyu2_test.csv', required=False, help='the path of testlist')
    parser.add_argument('--root_path', required=False,default='./', help="the root path of dataset")
    parser.add_argument('--batch_size', type=int, default=24, help='training batch size')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--lr', default=0.0002, type=float, help='initial learning rate')
    parser.add_argument('--resume', action='store_true',default=False, help='continue training the model')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum parameter used in the Optimizer.')
    parser.add_argument('--epsilon', default=0.001, type=float, help='epsilon')
    parser.add_argument('--optimizer_name', default="adam", type=str, help="Optimizer selection")
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--loadckpt', default=None, help="Specify the path to a specific model")
    parser.add_argument('--pretrained_dir', required=False,default='./pretrained',type=str, help="the path of pretrained models")
    parser.add_argument("--notes", default='', type=str, help="Wandb notes")
    parser.add_argument("--tags", default='sweep', type=str, help="Wandb tags")
    parser.add_argument("--name", default="DANet")
    parser.add_argument('--seed', '--rs', default=12, type=int,help='random seed (default: 0)')
    parser.add_argument('--validate-every', '--validate_every', default=500, type=int, help='validation period')
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)
    parser.add_argument('--height', type=float, help='feature height', default=8)
    parser.add_argument('--width', type=float, help='feature width', default=10)
    parser.add_argument('--w_chamfer', '--w-chamfer', default=0.1, type=float, help="weight value for chamfer loss")
    parser.add_argument('--w_minmax', '--w-minmax', default=0.1, type=float, help="weight value for minmax loss")
    parser.add_argument('--valley_factor', '--valley_factor', default=2, type=float, help="weight value for valley loss")
    return parser.parse_args()

