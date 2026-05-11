# -*- coding: utf-8 -*-
"""
KG-STGAT Main Program (overwrite version)
- Keeps entrypoint name main.py so you can run:
  python main.py --mode train --route_id 1
"""
import os
import sys
import random
import numpy as np
import torch

from config import get_config, print_config
from data_loader import create_dataloaders
from model import create_model
from trainer import train_model, test_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}\n")


def main():
    print("Loading configuration...\n")
    config = get_config()
    print_config(config)

    set_seed(config.seed)

    if config.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU instead")
        config.device = "cpu"
    print(f"Using device: {config.device}\n")

    print("=" * 70)
    print("TRAIN MODE" if config.mode == "train" else ("TEST MODE" if config.mode == "test" else "TRAIN AND TEST MODE"))
    print("=" * 70 + "\n")

    print("Creating data loaders...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(config, config.route_id)
        print("✓ Data loaders created successfully")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}\n")
    except Exception as e:
        print(f"✗ Failed to create data loaders: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Creating model...")
    try:
        model = create_model(config).to(config.device)
        print("✓ Model created successfully\n")
        if config.print_model:
            print(model)
            return
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return

    trained_model = None
    if config.mode in ["train", "both"]:
        print("=" * 70)
        print("Starting training...")
        print("=" * 70 + "\n")
        try:
            trained_model = train_model(model, train_loader, val_loader, config)
            print("\n✓ Training completed successfully!")
        except KeyboardInterrupt:
            print("\n⚠ Training interrupted by user")
            save_path = os.path.join(config.save_path, "interrupted_model.pth")
            torch.save({"epoch": 0, "model_state_dict": model.state_dict(), "val_loss": None}, save_path)
            print(f"  Model saved to: {save_path}")
            return
        except Exception as e:
            print(f"\n✗ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return

    if config.mode in ["test", "both"]:
        print("\n" + "=" * 70)
        print("Starting testing...")
        print("=" * 70 + "\n")

        if config.mode == "both" and trained_model is not None:
            model = trained_model
        elif config.mode == "test" and config.checkpoint:
            print(f"Loading checkpoint: {config.checkpoint}")
            try:
                ckpt = torch.load(config.checkpoint, map_location=config.device)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"])
                    print("✓ Checkpoint loaded successfully")
                    print(f"  Trained for: {ckpt.get('epoch', 'unknown')} epochs")
                    print(f"  Validation loss: {ckpt.get('val_loss', 'unknown')}\n")
                else:
                    model.load_state_dict(ckpt)
                    print("✓ Checkpoint loaded successfully (legacy)\n")
            except Exception as e:
                print(f"✗ Failed to load checkpoint: {e}")
                import traceback
                traceback.print_exc()
                return

        try:
            test_model(model, test_loader, config)
            print("\n✓ Testing completed successfully!")
        except Exception as e:
            print(f"\n✗ Testing failed: {e}")
            import traceback
            traceback.print_exc()
            return

    print("\n" + "=" * 70)
    print("ALL TASKS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Program failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
