import sys
sys.path.append('../../')
import argparse
import logging
from pathlib import Path

import torch
from amulet.defenses import AdversarialTraining
from amulet.risks import Poisoning
from amulet.utils import (
    load_data, 
    initialize_model, 
    train_classifier, 
    create_dir,
    get_accuracy
)

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
    parser.add_argument('--epsilon', type = int, default = 32, help = 'Controls amount of perturbations.')
    parser.add_argument('--poisoned_portion', type=float, default=0.2, help='posioning portion (float, range from 0 to 1, default: 0.1)')
    parser.add_argument('--trigger_label', type=int, default=0, help='The NO. of trigger label (int, range from 0 to 10, default: 0)')

    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    # Setup logger
    root_dir = Path(args.root)
    log_dir = root_dir / 'logs' 
    create_dir(log_dir)
    logging.basicConfig(level=logging.INFO, 
                        filename=log_dir / 'adversarial_training_poisoning.log', 
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
    target_model_path = models_path / 'target'
    target_model_filename = target_model_path / filename

    # Train or Load Target Model
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

    # Train or load model with Adversarial Training 
    defended_model_path = models_path / 'adversarial_training' / f'epsilon_{args.epsilon}'
    defended_model_filename = defended_model_path / filename
    
    if defended_model_filename.exists():
        log.info("Defended model loaded from %s", defended_model_filename)
        defended_model = torch.load(defended_model_filename)
    else:
        log.info("Retraining Model with Adversarial Training")
        adv_training = AdversarialTraining(target_model,
                                           criterion,
                                           optimizer,
                                           train_loader,
                                           args.device,
                                           args.epochs,
                                           args.epsilon)
        defended_model = adv_training.train_model()

        # Save model
        create_dir(defended_model_path, log)
        torch.save(defended_model, defended_model_filename)
        
    test_accuracy_defended = get_accuracy(defended_model, test_loader, args.device)
    log.info("Test accuracy of defended model: %s", test_accuracy_defended)      



    attack_model_path = models_path / 'model_poisoning' / f'poisoned_protion_{args.poisoned_portion}'
    attack_model_filename = attack_model_path / filename
    if attack_model_filename.exists():
        log.info("Attack model loaded from %s", attack_model_filename)
        attack_model = torch.load(attack_model_filename)
        model_poison = Poisoning(target_model,
                                           attack_model,
                                           optimizer,
                                           criterion,
                                           data.train_set.dataset,
                                           data.test_set,
                                           args.batch_size,
                                           args.trigger_label,
                                           args.poisoned_portion,
                                           args.device,
                                           args.dataset,
                                           args.epochs)
    else:
        log.info("Running Model Poisoning attack")
        attack_model = initialize_model(args.model, args.model_capacity, args.dataset, log).to(args.device)
        attack_model.load_state_dict(target_model.state_dict())
        optimizer = torch.optim.Adam(attack_model.parameters(), lr=1e-3)
        model_poison = Poisoning(target_model,
                                           attack_model,
                                           optimizer,
                                           criterion,
                                           data.train_set.dataset,
                                           data.test_set,
                                           args.batch_size,
                                           args.trigger_label,
                                           args.poisoned_portion,
                                           args.device,
                                           args.dataset,
                                           args.epochs)
        attack_model = model_poison.train_poisoned_model()

        # Save model
        create_dir(attack_model_path, log)
        torch.save(attack_model, attack_model_filename)
    
    # Use evaluate_attack() as a static method since the model might be loaded from file
    evaluation_results = model_poison.test_poisoned_model()
    log.info("Target Model on Origin Data: %s", evaluation_results['target_model_ori_test'])
    log.info("Target Model on Trigger Data: %s", evaluation_results['target_model_tri_test'])
    log.info("Attack Model on Origin Data: %s", evaluation_results['attack_model_ori_test'])
    log.info("Attack Model on Trigger Data: %s", evaluation_results['attack_model_tri_test'])
    
    attack_model_path = models_path / 'model_poisoning' / f'poisoned_protion_{args.poisoned_portion}'/'adversarial_trained'
    attack_model_filename = attack_model_path / filename
    if attack_model_filename.exists():
        log.info("Attack model loaded from %s", attack_model_filename)
        attack_model = torch.load(attack_model_filename)
        model_poison = Poisoning(defended_model,
                                           attack_model,
                                           optimizer,
                                           criterion,
                                           data.train_set.dataset,
                                           data.test_set,
                                           args.batch_size,
                                           args.trigger_label,
                                           args.poisoned_portion,
                                           args.device,
                                           args.dataset,
                                           args.epochs)
    else:
        log.info("Running Model Poisoning attack")
        attack_model = initialize_model(args.model, args.model_capacity, args.dataset, log).to(args.device)
        attack_model.load_state_dict(defended_model.state_dict())
        optimizer = torch.optim.Adam(attack_model.parameters(), lr=1e-3)
        model_poison = Poisoning(defended_model,
                                           attack_model,
                                           optimizer,
                                           criterion,
                                           data.train_set.dataset,
                                           data.test_set,
                                           args.batch_size,
                                           args.trigger_label,
                                           args.poisoned_portion,
                                           args.device,
                                           args.dataset,
                                           args.epochs)
        attack_model = model_poison.train_poisoned_model()

        # Save model
        create_dir(attack_model_path, log)
        torch.save(attack_model, attack_model_filename)
    
    # Use evaluate_attack() as a static method since the model might be loaded from file
    evaluation_results = model_poison.test_poisoned_model()
    log.info("Defended Model on Origin Data: %s", evaluation_results['target_model_ori_test'])
    log.info("Defended Model on Trigger Data: %s", evaluation_results['target_model_tri_test'])
    log.info("Attack Model on Origin Data: %s", evaluation_results['attack_model_ori_test'])
    log.info("Attack Model on Trigger Data: %s", evaluation_results['attack_model_tri_test'])
    
      

if __name__ == '__main__':
    args = parse_args()
    main(args)