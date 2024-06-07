import sys
sys.path.append('../../')
import argparse
import logging
from pathlib import Path

import torch
from amulet.defenses import GroupFairness
from amulet.risks import ModelExtraction
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
    parser.add_argument('--dataset', type = str, default = 'census', help = 'Options: lfw, census.')
    parser.add_argument('--model', type = str, default = 'binarynet', help = 'Options: binarynet.')
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
    parser.add_argument('--adv_train_fraction', 
                        type = float, 
                        default = 0.5, 
                        help = 'Fraction of trianing data used by the adversary.')
    
    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    # Setup logger
    root_dir = Path(args.root)
    log_dir = root_dir / 'logs' 
    create_dir(log_dir)
    logging.basicConfig(level=logging.INFO, 
                        filename=log_dir / 'group_fairness.log', 
                        filemode="w")
    log = logging.getLogger('All')
    log.addHandler(logging.StreamHandler())

    # Set random seeds for reproducibility
    torch.manual_seed(args.exp_id)
    generator = torch.Generator().manual_seed(args.exp_id)

    # Load dataset and create data loaders
    x_train, x_test, y_train, y_test, z_train, z_test = load_data(root_dir, 
                                                                  generator, 
                                                                  args.dataset, 
                                                                  args.training_size, 
                                                                  log,
                                                                  return_x_y_z = True)

    train_set = torch.utils.data.TensorDataset(torch.from_numpy(x_train).type(torch.FloatTensor),
                                               torch.from_numpy(y_train).type(torch.LongTensor))
    test_set = torch.utils.data.TensorDataset(torch.from_numpy(x_test).type(torch.FloatTensor),
                                              torch.from_numpy(y_test).type(torch.LongTensor))

    train_set_wsattr = torch.utils.data.TensorDataset(torch.from_numpy(x_train).type(torch.FloatTensor),
                                               torch.from_numpy(y_train).type(torch.LongTensor),
                                               torch.from_numpy(z_train).type(torch.FloatTensor))
    test_set_wsattr = torch.utils.data.TensorDataset(torch.from_numpy(x_test).type(torch.FloatTensor),
                                              torch.from_numpy(y_test).type(torch.LongTensor),
                                              torch.from_numpy(z_test).type(torch.FloatTensor))


    adv_train_size = int(args.adv_train_fraction*len(train_set))
    target_train_size = len(train_set) - adv_train_size
    target_train_set, adv_train_set = torch.utils.data.random_split(train_set, [target_train_size, adv_train_size], generator=generator)
    target_train_set_wsattr, _ = torch.utils.data.random_split(train_set_wsattr, [target_train_size, adv_train_size], generator=generator)

    # Create data loaders
    adv_train_loader = torch.utils.data.DataLoader(dataset=adv_train_set, batch_size=args.batch_size, shuffle=False)
    target_train_loader = torch.utils.data.DataLoader(dataset=target_train_set, batch_size=args.batch_size, shuffle=False)
    target_train_loader_wsattr = torch.utils.data.DataLoader(dataset=target_train_set_wsattr, batch_size=args.batch_size, shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)
    test_loader_wsattr = torch.utils.data.DataLoader(dataset=test_set_wsattr, batch_size=args.batch_size, shuffle=False)

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
        target_model = train_classifier(target_model, target_train_loader, criterion, optimizer, args.epochs, args.device)
        log.info("Target model trained")

        # Save model
        create_dir(target_model_path, log)
        torch.save(target_model, target_model_filename)

    test_accuracy_target = get_accuracy(target_model, test_loader, args.device)
    log.info("Test accuracy of target model: %s", test_accuracy_target)      

    # Train or load model with Adversarial Training 
    defended_model_path = models_path / 'group_fairness' 
    defended_model_filename = defended_model_path / filename
    
    if defended_model_filename.exists():
        log.info("Defended model loaded from %s", defended_model_filename)
        defended_model = torch.load(defended_model_filename)
    else:
        log.info("Retraining Model with Group Fairness")
        if args.dataset=="LFW": # change lambdas manually to get better trade-off; hyperparameter tuning is hard
            lambdas = torch.Tensor([45, 17])  
        else: 
            lambdas = torch.Tensor([40, 40])
        group_fairness = GroupFairness(target_model,
                                           criterion,
                                           optimizer,
                                           target_train_loader_wsattr,
                                           test_loader_wsattr,
                                           lambdas,
                                           args.device,
                                           args.epochs)
        defended_model, old_metrics= group_fairness.train_fair()
        new_metrics = group_fairness.generate_report()
        print(old_metrics,new_metrics)
        
        # Save model
        create_dir(defended_model_path, log)
        torch.save(defended_model, defended_model_filename)
        
    test_accuracy_defended = get_accuracy(defended_model, test_loader, args.device)
    log.info("Test accuracy of defended model: %s", test_accuracy_defended)      
    

    # Model Extraction on target model
    attack_model_path = models_path / 'model_extraction' / f'adv_train_fraction_{args.adv_train_fraction}'
    attack_model_filename = attack_model_path / filename

    if attack_model_filename.exists():
        log.info("Attack model loaded from %s", attack_model_filename)
        attack_model = torch.load(attack_model_filename)
    else:
        log.info("Running Model Extraction attack")
        attack_model = initialize_model(args.model, args.model_capacity, args.dataset, log).to(args.device)
        model_extraction = ModelExtraction(target_model,
                                           attack_model,
                                           optimizer,
                                           criterion,
                                           adv_train_loader,
                                           args.device,
                                           args.epochs)
        attack_model = model_extraction.train_attack_model()

        # Save model
        create_dir(attack_model_path, log)
        torch.save(attack_model, attack_model_filename)
    
    # Use evaluate_attack() as a static method since the model might be loaded from file
    evaluation_results_target = ModelExtraction.evaluate_attack(target_model, attack_model, test_loader, args.device)

    log.info("Target Model accuracy: %s", evaluation_results_target['target_accuracy'])
    log.info("Stolen-from-target Model accuracy: %s", evaluation_results_target['stolen_accuracy'])
    log.info("Fidelity: %s", evaluation_results_target['fidelity'])
    log.info("Correct Fidelity: %s", evaluation_results_target['correct_fidelity']) 

    # Model Extraction on defended model 
    attack_model_2_path = models_path / 'group_fairness' / 'model_extraction' / f'adv_train_fraction_{args.adv_train_fraction}'
    attack_model_2_filename = attack_model_2_path / filename

    if attack_model_2_filename.exists():
        log.info("Attack model loaded from %s", attack_model_2_filename)
        attack_model_2 = torch.load(attack_model_2_filename)
    else:
        log.info("Running Model Extraction attack")
        attack_model_2 = initialize_model(args.model, args.model_capacity, args.dataset, log).to(args.device)
        model_extraction = ModelExtraction(defended_model,
                                           attack_model_2,
                                           optimizer,
                                           criterion,
                                           adv_train_loader,
                                           args.device,
                                           args.epochs)
        attack_model_2 = model_extraction.train_attack_model()

        # Save model
        create_dir(attack_model_2_path, log)
        torch.save(attack_model_2, attack_model_2_filename)
    
    # Use evaluate_attack() as a static method since the model might be loaded from file
    evaluation_results_defended = ModelExtraction.evaluate_attack(defended_model, attack_model_2, test_loader, args.device)

    log.info("Defended Model accuracy: %s", evaluation_results_defended['target_accuracy'])
    log.info("Stolen-from-defended Model accuracy: %s", evaluation_results_defended['stolen_accuracy'])
    log.info("Fidelity: %s", evaluation_results_defended['fidelity'])
    log.info("Correct Fidelity: %s", evaluation_results_defended['correct_fidelity'])     

if __name__ == '__main__':
    args = parse_args()
    main(args)