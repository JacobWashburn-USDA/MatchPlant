"""
Script Name: transfer_train.py
Purpose: Main script for transfer learning with Faster R-CNN
Authors: Worasit Sangjan
Date: 15 February 2025 
Version: 1.0
"""

import torch
from pathlib import Path
from train import (
    MaizeDatasetCOCO, 
    build_model, 
    get_transform, 
    configure_device_and_resources,
    evaluate,
    train_one_epoch
)
from train_config_loader import ConfigLoader
from transfer_utils import (
    load_pretrained_model,
    freeze_layers,
    get_transfer_optimizer,
    print_trainable_parameters
)
from torch.utils.data import DataLoader

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # Load configurations
    config_loader = ConfigLoader()
    config_dict = config_loader.load_config('transfer_config.yaml')
    model_config, training_config, data_config, resource_config = config_loader.create_configs(config_dict)
    transfer_config = config_dict.get('transfer', {})
    
    # Configure device
    device, num_workers = configure_device_and_resources()
    
    # Initialize model
    print("Initializing model...")
    model = build_model(model_config)
    
    # Load pretrained weights
    print(f"Loading pretrained weights from {transfer_config['pretrained_model_path']}")
    checkpoint = load_pretrained_model(model, transfer_config['pretrained_model_path'])
    
    # Freeze layers according to config
    print("Applying transfer learning settings...")
    freeze_layers(model, transfer_config)
    
    # Move model to device
    model.to(device)
    
    # Print parameter summary
    print_trainable_parameters(model)
    
    # Create datasets
    print("Creating datasets...")
    dataset = MaizeDatasetCOCO(
        Path(data_config.train_dir),
        Path(data_config.train_annotations),
        transforms=get_transform(train=True, aug_config=training_config.augmentation)
    )
    
    dataset_val = MaizeDatasetCOCO(
        Path(data_config.val_dir),
        Path(data_config.val_annotations),
        transforms=get_transform(train=False)
    )
    
    # Create data loaders
    print("Creating data loaders...")
    data_loader = DataLoader(
        dataset,
        batch_size=transfer_config['training']['batch_size'],
        shuffle=True,
        num_workers=data_config.num_workers,
        collate_fn=collate_fn,
        pin_memory=data_config.pin_memory,
        persistent_workers=data_config.persistent_workers
    )
    
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=transfer_config['training']['batch_size'],
        shuffle=False,
        num_workers=data_config.num_workers,
        collate_fn=collate_fn,
        pin_memory=data_config.pin_memory,
        persistent_workers=data_config.persistent_workers
    )
    
    # Create optimizer with transfer learning settings
    optimizer = get_transfer_optimizer(model, transfer_config)
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1
    )
    
    # Setup checkpointing
    checkpoint_dir = Path(training_config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Training loop
    print("Starting transfer learning...")
    try:
        best_map = 0.0
        start_epoch = transfer_config['training'].get('initial_epoch', 0)
        num_epochs = transfer_config['training']['epochs']
        
        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 20)
            
            # Train one epoch
            train_loss = train_one_epoch(model, optimizer, data_loader, device, epoch)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Evaluate
            val_stats = evaluate(model, data_loader_val, device, epoch=epoch, train_loss=train_loss)
            val_map = val_stats[1]  # mAP at IoU=0.50
            print(f"Validation mAP: {val_map:.4f}")
            
            # Update learning rates
            lr_scheduler.step()
            
            # Save best model
            if val_map > best_map:
                best_map = val_map
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_map': best_map,
                }, checkpoint_dir / 'best_model_transfer.pt')
            
            # Regular checkpoints
            if epoch % training_config.save_freq == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_map': val_map,
                }, checkpoint_dir / f'model_transfer_epoch_{epoch}.pt')
                
    except Exception as e:
        print(f"Training error: {e}")
        # Save emergency checkpoint
        torch.save(model.state_dict(), checkpoint_dir / 'emergency_save_transfer.pt')

if __name__ == "__main__":
    main()