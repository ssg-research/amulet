import sys
sys.path.append('../../')
import argparse
import logging
from pathlib import Path

import torch
from amulet.defenses import OutlierRemoval
from amulet.risks import DataReconstruction
from amulet.utils import (
    load_data, 
    initialize_model, 
    train_classifier, 
    create_dir,
    get_accuracy
)

# TODO: Got the following error when running this code:
# Traceback (most recent call last):
#  File "/home/a7waheed/conflict-library/examples/interactions/out_rem_data_recon.py", line 147, in <module>
#    main(args)
#  File "/home/a7waheed/conflict-library/examples/interactions/out_rem_data_recon.py", line 109, in main
#    defended_model = outlier_removal.train_model()
#                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#  File "/home/a7waheed/conflict-library/examples/interactions/../../mlconf/defenses/outlier_removal.py", line 132, in train_model
#    train_data_new = torch.utils.data.TensorDataset(torch.from_numpy(np.array(train_inputs_new)).type(torch.FloatTensor),
#                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# TypeError: can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', 
                        type = str, 
                        default = '../../', 
                        help='Root directory of models and datasets.')
    parser.add_argument('--dataset', type = str, default = 'cifar10', help = 'Options: cifar10, fmnist, lfw, census.')
    parser.add_argument('--model', type = str, default = 'vgg', help = 'Options: vgg, linearnet, binarynet.')
    parser.add_argument('--model_capacity', 
                        type = str, 
                        default = 'm1', 
                        help = 'Size of the model to use. Options: m1, m2, m3, m4, where m1 is the smallest.')
    parser.add_argument('--training_size', type = float, default = 1, help = 'Fraction of dataset to use.')
    parser.add_argument('--batch_size', type = int, default = 256, help = 'Batch size of input data.')
    parser.add_argument('--epochs', type = int, default = 1, help = 'Number of epochs for training.')
    parser.add_argument('--device', 
                        type = str, 
                        default = torch.device('cuda:{0}'.format(0) if torch.cuda.is_available() else 'cpu'), 
                        help = 'Device on which to run PyTorch')
    parser.add_argument('--exp_id', type = int, default = 0, help = 'Used as a random seed for experiments.')
    parser.add_argument('--percent', type = int, default = 10, help = 'Percentage of outliers to remove.')
    parser.add_argument('--alpha', type = int, default = 3000, help = 'Number of iterations for data reconstruction.')

    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    # Setup logger
    root_dir = Path(args.root)
    log_dir = root_dir / 'logs' 
    create_dir(log_dir)
    logging.basicConfig(level=logging.INFO, 
                        filename=log_dir / 'outlier_removal_data_reconstruction.log', 
                        filemode="w")
    log = logging.getLogger('All')
    log.addHandler(logging.StreamHandler())

    # Set random seeds for reproducibility
    torch.manual_seed(args.exp_id)
    generator = torch.Generator().manual_seed(args.exp_id)

    # Load dataset and create data loaders
    data = load_data(root_dir, generator, args.dataset, args.training_size, log)
    train_loader = torch.utils.data.DataLoader(dataset=data.train_set, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=data.test_set, batch_size=args.batch_size, shuffle=False)

    # Set up filename and directories to save/load models
    models_path = root_dir / 'saved_models'
    filename = f'{args.dataset}_{args.model}_{args.model_capacity}_{args.training_size*100}_{args.batch_size}_{args.epochs}_{args.exp_id}.pt'

    # Train or Load Target Model
    target_model_path = models_path / 'target'
    target_model_filename = target_model_path / filename
    criterion = torch.nn.CrossEntropyLoss()
    
    if target_model_filename.exists():
        log.info("Target model loaded from %s", target_model_filename)
        target_model = torch.load(target_model_filename).to(args.device)
        optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-3)
    else:
        log.info("Training target model")
        target_model = initialize_model(args.model, args.model_capacity, args.dataset, log).to(args.device)
        optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-3)
        target_model = train_classifier(target_model, train_loader, criterion, optimizer, args.epochs, args.device)
        log.info("Target model trained")

        # Save model
        create_dir(target_model_path, log)
        torch.save(target_model, target_model_filename)

    test_accuracy_target = get_accuracy(target_model, test_loader, args.device)
    log.info("Test accuracy of target model: %s", test_accuracy_target)      

    # Train or Load model with Outlier Removal
    defended_model_path = models_path / 'outlier_removal' / f'percent_{args.percent}'
    defended_model_filename = defended_model_path / filename

    if defended_model_filename.exists():
        log.info("Defended model loaded from %s", defended_model_filename)
        defended_model = torch.load(defended_model_filename)
    else:
        log.info("Retraining Model after Outlier Removal")
        outlier_removal = OutlierRemoval(target_model,
                                         criterion,
                                         optimizer,
                                         train_loader,
                                         test_loader,
                                         args.device,
                                         epochs=args.epochs,
                                         batch_size=args.batch_size,
                                         percent=args.percent)
        defended_model = outlier_removal.train_model()

        # Save model
        create_dir(defended_model_path, log)
        torch.save(defended_model, defended_model_filename)
    
    test_accuracy_outlier_removed = get_accuracy(defended_model, test_loader, args.device)
    log.info("Test accuracy of model with outliers removed %s", test_accuracy_outlier_removed)

    # Set batch size to 1 for data reconstruction
    recon_test_loader = torch.utils.data.DataLoader(dataset=data.test_set, batch_size=1, shuffle=False)

    # Data Reconstruction on target model
    log.info('Running Data Reconstruction Attack on target model')

    input_size = (1,) + tuple(data.test_set[0][0].shape)
    num_classes_dict = {'cifar10':10, 'fmnist': 10, 'census': 2, 'lfw': 2}
    output_size = num_classes_dict[args.dataset]
    data_recon = DataReconstruction(target_model, 
                                   input_size,
                                   output_size,
                                   args.device,
                                   args.alpha)

    test_loss = data_recon.reverse_mse(recon_test_loader)
    log.info(f'MSE Loss on test dataset (target model): {test_loss:.4f}')

    # Data Reconstruction on defended model 
    log.info('Running Data Reconstruction Attack on defended model')

    data_recon = DataReconstruction(defended_model, 
                                   input_size,
                                   output_size,
                                   args.device,
                                   args.alpha)

    test_loss = data_recon.reverse_mse(recon_test_loader)
    log.info(f'MSE Loss on test dataset (defended model): {test_loss:.4f}')

if __name__ == '__main__':
    args = parse_args()
    main(args)