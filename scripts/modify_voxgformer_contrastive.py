#!/usr/bin/env python3
"""
Correctly add contrastive loss to VoxGFormer.py
"""

import re

def modify_voxgformer():
    with open('VoxGFormer.py', 'r') as f:
        content = f.read()
    
    # 1. Add import
    if 'from losses.contrastive_loss import' not in content:
        content = content.replace(
            'from check_gpu_memory import print_gpu_memory_usage, print_tensor_memory, clear_gpu_memory',
            'from losses.contrastive_loss import GraphContrastiveLoss\nfrom check_gpu_memory import print_gpu_memory_usage, print_tensor_memory, clear_gpu_memory'
        )
    
    # 2. Add contrastive loss initialization AFTER self.to(self.device)
    # This ensures the module is on the correct device
    init_code = '''
        # 对比学习模块 (在 self.to() 之后初始化以确保设备正确)
        self.use_contrastive = getattr(args, 'use_contrastive', False)
        if self.use_contrastive:
            self.contrastive_loss_fn = GraphContrastiveLoss(
                hidden_dim=args.embedding_dim,
                temperature=getattr(args, 'contrastive_temp', 0.1),
                aug_ratio=getattr(args, 'contrastive_aug_ratio', 0.2)
            ).to(self.device)
        else:
            self.contrastive_loss_fn = None

'''
    
    # Find where to insert (after self.to(self.device))
    pattern = r'(self\.to\(self\.device\)\s*\n)(\s*\n\s*def TransformerEncoder)'
    content = re.sub(pattern, r'\1' + init_code + r'\2', content)
    
    # 3. Add contrastive loss computation in forward (inside train_flag=True block)
    # Find the loss_ring calculation and add after it
    contrastive_code = '''
            # 对比学习损失
            loss_contrastive = torch.tensor(0.0, device=emb.device)
            if self.use_contrastive and self.contrastive_loss_fn is not None:
                if normal_for_train_idx is not None and len(normal_for_train_idx) > 1:
                    loss_contrastive = self.contrastive_loss_fn(emb[0, normal_for_train_idx, :])

'''
    
    pattern = r'(loss_ring = torch\.mean\(ring_out_range_loss \+ ring_in_range_loss\)\s*\n)'
    content = re.sub(pattern, r'\1' + contrastive_code, content)
    
    # 4. Add loss_contrastive initialization in else branch (train_flag=False)
    # Find the else: after train_flag block and add initialization
    pattern = r'(\s+else:\s*\n)(\s+f_1 = self\.fc1\(emb\))'
    replacement = r'\1            loss_contrastive = torch.tensor(0.0, device=emb.device)\n\2'
    content = re.sub(pattern, replacement, content)
    
    # 5. Update return statement
    content = content.replace(
        'return emb, emb_combine, logits, outlier_emb, noised_normal_for_generation_emb, loss_rec, loss_ring',
        'return emb, emb_combine, logits, outlier_emb, noised_normal_for_generation_emb, loss_rec, loss_ring, loss_contrastive'
    )
    
    with open('VoxGFormer.py', 'w') as f:
        f.write(content)
    
    print('VoxGFormer.py modified successfully')

if __name__ == '__main__':
    modify_voxgformer()