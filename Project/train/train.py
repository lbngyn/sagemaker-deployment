#!/usr/bin/env python
import argparse
import json
import os
import pickle
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import LSTMClassifier

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(model_info['embedding_dim'], model_info['hidden_dim'], model_info['vocab_size'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Load the saved word_dict.
    word_dict_path = os.path.join(model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'rb') as f:
        model.word_dict = pickle.load(f)

    model.to(device).eval()

    print("Done loading model.")
    return model

def _get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")

    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    train_X = torch.from_numpy(train_data.drop([0], axis=1).values).long()

    train_ds = torch.utils.data.TensorDataset(train_X, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

def _get_test_data_loader(batch_size, training_dir):
    """Get test data loader."""
    print("Get test data loader.")

    test_data = pd.read_csv(os.path.join(training_dir, "test.csv"), header=None, names=None)
    
    test_y = torch.from_numpy(test_data[[0]].values).float().squeeze()
    test_X = torch.from_numpy(test_data.drop([0], axis=1).values).long()

    test_ds = torch.utils.data.TensorDataset(test_X, test_y)
    return torch.utils.data.DataLoader(test_ds, batch_size=batch_size)


def train(model, train_loader, epochs, optimizer, loss_fn, device, use_sigmoid=True):
    """Train the model and return metrics for each epoch."""
    training_metrics = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        predictions = []
        true_labels = []
        
        for batch in train_loader:         
            batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_X)
            
            if use_sigmoid:
                output = torch.sigmoid(output)
            
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            pred = (output > 0.5).float()
            predictions.extend(pred.cpu().numpy())
            true_labels.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        
        metrics = {
            'epoch': epoch,
            'train_loss': avg_loss,
            'train_accuracy': accuracy,
            'train_precision': precision,
            'train_recall': recall,
            'train_f1': f1
        }
        
        training_metrics.append(metrics)
        print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    return training_metrics

def evaluate(model, test_loader, device, loss_fn):
    """Evaluate the model on test data."""
    model.eval()
    test_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            output = model(batch_X)
            output = torch.sigmoid(output)
            
            test_loss += loss_fn(output, batch_y).item()
            
            pred = (output > 0.5).float()
            predictions.extend(pred.cpu().numpy())
            true_labels.extend(batch_y.cpu().numpy())
    
    # Calculate metrics
    test_loss = test_loss / len(test_loader)
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    metrics = {
        'test_loss': test_loss,
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1
    }
    
    return metrics

if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Model Parameters
    parser.add_argument('--embedding_dim', type=int, default=32, metavar='N',
                        help='size of the word embeddings (default: 32)')
    parser.add_argument('--hidden_dim', type=int, default=100, metavar='N',
                        help='size of the hidden dimension (default: 100)')
    parser.add_argument('--vocab_size', type=int, default=5000, metavar='N',
                        help='size of the vocabulary (default: 5000)')

    # SageMaker Parameters
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))

    # S3 paramenters 
    parser.add_argument('--bucket_name', type=str, required=True, help='Name of the S3 bucket to store data')
    parser.add_argument('--prefix', type=str, required=True, help='Prefix (folder path) within the S3 bucket')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load both training and test data
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)
    test_loader = _get_test_data_loader(args.batch_size, args.data_dir)

    # Build and train model
    model = LSTMClassifier(args.embedding_dim, args.hidden_dim, args.vocab_size).to(device)
    
    with open(os.path.join(args.data_dir, "word_dict.pkl"), "rb") as f:
        model.word_dict = pickle.load(f)

    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.BCELoss()

    # Train and get training metrics
    training_metrics = train(model, train_loader, args.epochs, optimizer, loss_fn, device)
    
    # Evaluate on test set
    test_metrics = evaluate(model, test_loader, device, loss_fn)
    
    # Save all metrics to report.csv
    all_metrics = []
    for epoch_metrics in training_metrics:
        metrics_row = {
            'epoch': epoch_metrics['epoch'],
            'split': 'train',
            'loss': epoch_metrics['train_loss'],
            'accuracy': epoch_metrics['train_accuracy'],
            'precision': epoch_metrics['train_precision'],
            'recall': epoch_metrics['train_recall'],
            'f1': epoch_metrics['train_f1']
        }
        all_metrics.append(metrics_row)
    
    # Add test metrics
    test_metrics_row = {
        'epoch': 'final',
        'split': 'test',
        'loss': test_metrics['test_loss'],
        'accuracy': test_metrics['test_accuracy'],
        'precision': test_metrics['test_precision'],
        'recall': test_metrics['test_recall'],
        'f1': test_metrics['test_f1']
    }
    all_metrics.append(test_metrics_row)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(f's3://{args.bucket_name}/{args.prefix}/reports.csv', index=False)
    

    # Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'vocab_size': args.vocab_size,
        }
        torch.save(model_info, f)

	# Save the word_dict
    word_dict_path = os.path.join(args.model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'wb') as f:
        pickle.dump(model.word_dict, f)

	# Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
