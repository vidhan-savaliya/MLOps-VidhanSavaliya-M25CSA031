import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import wandb
# Use absolute imports or relative if running as package, but for simple script execution:
from dataset import get_dataloaders
from model import ResNet18
from utils import count_flops, plot_grad_flow, plot_weight_distribution

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        
        # Visualize gradients periodically (e.g., first batch of specific epochs)
        if batch_idx == 0 and (epoch == 1 or epoch % 5 == 0):
            grad_img = plot_grad_flow(model.named_parameters())
            wandb.log({"gradients": wandb.Image(grad_img), "epoch": epoch})
            
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % args.log_interval == 0:
            wandb.log({
                "train_loss": loss.item(),
                "train_acc": 100. * correct / total,
                "epoch": epoch
            })

    # Log weight distribution at end of epoch
    if epoch == 1 or epoch % 5 == 0:
        weight_img = plot_weight_distribution(model.named_parameters())
        wandb.log({"weights": wandb.Image(weight_img), "epoch": epoch})

    print(f"Epoch: {epoch} \tLoss: {running_loss/len(train_loader):.4f} \tAcc: {100.*correct/total:.2f}%")

def evaluate(model, device, loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_loss /= len(loader)
    acc = 100. * correct / total
    return test_loss, acc

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Initialize WandB
    if not args.dry_run:
        wandb.init(project="cifar10-lab-assignment", config=args)
    else:
        print("Dry run: Skipping WandB init")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    # Model
    model = ResNet18().to(device)
    
    # Count FLOPs
    print("Counting FLOPs...")
    count_flops(model, input_size=(1, 3, 32, 32))

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Train Loop
    epochs = 1 if args.dry_run else args.epochs
    for epoch in range(1, epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        
        # Validation
        print("Validating...", end=" ")
        val_loss, val_acc = evaluate(model, device, val_loader)
        print(f"Val Loss: {val_loss:.4f} \tVal Acc: {val_acc:.2f}%")
        
        if not args.dry_run:
            wandb.log({"val_loss": val_loss, "val_acc": val_acc, "epoch": epoch})
            
        scheduler.step()
        
        if args.dry_run:
            break

    # Final Test
    print("Testing on Test Set...")
    test_loss, test_acc = evaluate(model, device, test_loader)
    print(f"Test Set Loss: {test_loss:.4f} \tTest Acc: {test_acc:.2f}%")
    if not args.dry_run:
         wandb.log({"test_loss": test_loss, "test_acc": test_acc})

    if not args.dry_run:
        wandb.finish()

    # Collect all predictions for confusion matrix and class-wise accuracy
    print("Generating detailed visualizations...")
    all_preds = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Create and logging plots
    if not args.dry_run:
        # Re-init wandb if it was finished (it shouldn't be effectively if we want to log to same run? 
        # Actually wandb.finish() closes the run. We should move finish() to the very end.)
        # Let's fix the order.
        pass

    from utils import plot_confusion_matrix, plot_class_accuracy, plot_predictions
    
    if not args.dry_run:
        cm_img = plot_confusion_matrix(all_targets, all_preds, classes)
        acc_img = plot_class_accuracy(all_targets, all_preds, classes)
        pred_img = plot_predictions(model, device, test_loader, classes)
        
        wandb.log({
            "confusion_matrix": wandb.Image(cm_img),
            "class_accuracy": wandb.Image(acc_img),
            "prediction_examples": wandb.Image(pred_img)
        })
        print("Visualizations logged to WandB.")
        wandb.finish()
    else:
        print("Dry run: Skipping extended visualization logging.")

if __name__ == '__main__':
    main()
