#!/usr/bin/env python3
"""
Video Segmentation Project - Main Entry Point
Usage:
    python main.py --model efficienttam --mode train
    python main.py --model nanovlm --mode train
    python main.py --model smolvlm --mode train
"""

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Video Segmentation Project')
    parser.add_argument('--model', type=str, choices=['efficienttam', 'nanovlm', 'smolvlm'], 
                       required=True, help='Model to run')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], 
                       default='train', help='Mode: train or test')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    
    args = parser.parse_args()
    
    # مسیر config
    if args.config is None:
        config_path = f'configs/{args.model}_config.yaml'
    else:
        config_path = args.config
    
    # اجرای مدل
    if args.model == 'efficienttam':
        from models.efficienttam.train import train
    elif args.model == 'nanovlm':
        from models.nanovlm.train import train
    elif args.model == 'smolvlm':
        from models.smolvlm.train import train
    
    train(config_path)
    print(f"✅ {args.model} {args.mode} completed!")

if __name__ == "__main__":
    main()
