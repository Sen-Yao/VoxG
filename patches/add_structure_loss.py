#!/usr/bin/env python3
"""Add structure loss to run.py"""

import re

def add_structure_loss():
    with open('run.py', 'r') as f:
        content = f.read()
    
    # 1. Add import after existing imports
    if 'from structure_reconstruction import' not in content:
        content = content.replace(
            'from sklearn.metrics import average_precision_score',
            'from sklearn.metrics import average_precision_score\nfrom structure_reconstruction import compute_degree_loss, compute_structure_loss'
        )
    
    # 2. Add command line arguments before the contrastive args
    args_code = '''
    # 结构重构参数
    parser.add_argument('--structure_loss_weight', type=float, default=0.0, help='Structure reconstruction loss weight')
    parser.add_argument('--structure_loss_type', type=str, default='degree', choices=['degree', 'edge'], help='Type of structure loss')

'''
    if '--structure_loss_weight' not in content:
        content = content.replace(
            '# 对比学习参数',
            args_code + '    # 对比学习参数'
        )
    
    with open('run.py', 'w') as f:
        f.write(content)
    
    print('run.py updated with structure loss args')

if __name__ == '__main__':
    add_structure_loss()
