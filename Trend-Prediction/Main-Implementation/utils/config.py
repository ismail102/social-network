import argparse

def get_arguments():

    parser = argparse.ArgumentParser()

    ## Optimizer
    parser.add_argument('--optimizer', type=str, default='adam', help="type \
                        of optimizer")

    parser.add_argument('--lr', type=float, default=3e-3,
                        help='learning rate')

    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.0)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Drop out')
    
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of training Epochs')
    parser.add_argument('--total_steps', type=int, default=-1, help='Total Step')
    
    parser.add_argument('--tokenizer_max_len', type=int, default=70,
                        help='Maxinum lenght of each token')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')

    parser.add_argument('--activation', type=str, default="relu",
                        help='SGD momentum (default: 0.0)')

    # other arguments
    parser.add_argument('--dataset', type=str, default='dr', help="name \
                        of dataset")

    parser.add_argument('--max_grad_norm', type=float, default= 1.0, help='DP MAX_GRAD_NORM')

    # Can be changed the following values for model training
    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--batch_size', type=int, default=8, help='local batch size')

    parser.add_argument('--model_ckpt', type=str, default='bert-base-uncased',
                            help='Image classification model name[Resnet50, Alexnet, Vgg16, Squeezenet1.1, Densenet121, #InceptionV3, #googelnet]')
    parser.add_argument('--gpu', default="cuda", help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")

    parser.add_argument('--data_set', type=int, default=-1, help="Select Dataset which will be trained.")
    parser.add_argument('--data_dir', type=str, default="/home/siu856533724/code/source-code/Social-Networks/Trend-Prediction/DataSet/csv")

    parser.add_argument('--train', type=str, default="/data/csv/train_labels_1.csv", help="The Label file for training dataset.")
    parser.add_argument('--test', type=str, default="/data/csv/train_labels_1.csv", help="The Label file for test dataset.")

    parser.add_argument('--train_images_npy', type=str, default='/data/train/dr_train_images_1_class2_', help='Converted data set')
    parser.add_argument('--train_labels_npy', type=str, default='/data/train/dr_train_labels_1_class2_', help='Converted data label')
    parser.add_argument('--test_images_npy', type=str, default='/data/test/dr_test_images_1_class2_', help='Converted data set')  
    parser.add_argument('--test_labels_npy', type=str, default='/data/test/dr_test_labels_1_class2_', help='Converted data label')

    parser.add_argument('--exp_name', type=str, default="Trend", help="The name of current experiment for logging.")
    parser.add_argument('--num_labels', type=int, default=5, help='how many classes have to be predicted.')
    args = parser.parse_args([])

    return args

if __name__ == '__main__':
    print("#Configuration")
    args = get_arguments()