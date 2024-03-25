import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from lib.modules.mlp import MLP

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def func_attention(query, context, region_lens, raw_feature_norm, smooth, eps=1e-8):
    """
    query(segmentation): (bz, 1, d)
    context(regions): (bz, rgn_num, d)
    """

    batch_size, rgn_num, embed_size = context.size(0), context.size(1), context.size(2)
    ### Mask
    pe_lens = region_lens.unsqueeze(1).repeat(1, rgn_num).to(region_lens.device)
    mask = torch.arange(rgn_num).expand(batch_size, rgn_num).to(region_lens.device)
    mask = (mask < pe_lens.long())
    pe_lens = pe_lens.masked_fill(mask == 0, 0)   
    
    # Get attention
    queryT = torch.transpose(query, 1, 2)   #(bz, d, 1)
    
    attn = torch.matmul(context, queryT) / np.sqrt(embed_size) # (bz, rgn_num, 1)
    attn = attn.view(batch_size,-1) # (bz, rgn_num*rgn_num)
    attn = attn.masked_fill(mask==0, -np.inf)
#     attn = F.softmax(attn*smooth, dim=-1)                #(n*qL, cL)
    attn = torch.sigmoid(attn)
#     for i in range(attn.shape[0]):
#         if attn[i].sum()<35:
#             print(attn[i])
#     import pdb
#     pdb.set_trace()
    weightedContext = attn.unsqueeze(-1)*context 
    
    del pe_lens
    return weightedContext, attn, mask

class View(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class CrossmodalFusion(nn.Module):
    def __init__(self, img_dim, embed_size, raw_feature_norm="clipped_l2norm", lambda_softmax=4.):
        super(CrossmodalFusion, self).__init__()
        self.raw_feature_norm = raw_feature_norm
        self.lambda_softmax = lambda_softmax
        
        ## MLP 
        self.MLP_image = MLP(embed_size, embed_size // 2, embed_size, 2) 
        self.MLP_seg = MLP(embed_size, embed_size // 2, embed_size, 2) 
        
        self.embed_seg_HW = nn.Sequential(
                # nn.AdaptiveAvgPool2d((7, 7)),
                nn.AvgPool2d(7), ## [bt, 133, 1,1]
                View(), ## [bt, 133]
                nn.Linear(133, embed_size),
                # nn.Sigmoid(),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.LayerNorm(embed_size))  
        
        self.fc_scale = nn.Linear(embed_size, embed_size)
#         self.fc_shift = nn.Linear(embed_size, embed_size)
        
#         self.embed_fusion = MLP(embed_size, embed_size // 2, embed_size, 2)  
        self.fc_1 = nn.Linear(embed_size, embed_size)
#         self.fc_2 = nn.Linear(embed_size, embed_size)
        
    def refine(self, seg, weiContext, mask):
        scaling = torch.tanh(self.fc_scale(weiContext))
        seg = seg.repeat(1,weiContext.size(1),1)*((mask*1).unsqueeze(-1))
#         shifting = torch.tanh(self.fc_shift(seg))
#         modu_res = self.embed_fusion(weiContext * scaling + seg)
        modu_res = F.relu(self.fc_1(weiContext * scaling + seg))
        return modu_res
    
    def forward(self, rgns, Unet_segs, region_lens):
        rgns = self.MLP_image(rgns)+ rgns # [bt, rgn_num, img_dim]-->[bt, rgn_num, embed_size]
        
        seg_features_Query = self.embed_seg_HW(Unet_segs)
        seg_features_Query = seg_features_Query.unsqueeze(1).contiguous() ## # [bt, embed_size]--> # [bt, 1, embed_size] 
        seg_features_Query = self.MLP_seg(seg_features_Query) + seg_features_Query # [bt, 133, 7, 7]--->[bt, embed_size]

        weirgns, _, mask = func_attention(seg_features_Query, rgns, region_lens, self.raw_feature_norm, smooth=self.lambda_softmax)
        ref_img = self.refine(seg_features_Query, weirgns, mask)
        
        del rgns, seg_features_Query
        return ref_img 
    
  
    
    

