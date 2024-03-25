import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math
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

def func_attention(query, context, raw_feature_norm, smooth, eps=1e-8):
    """
    query(rgns): (bz, reg_num, 128)
    context(seg_map): (bz, c=128, h, w)
    """
    batch_size, rgn_num, embed_size = query.size(0), query.size(1), query.size(2)
    context = context.view(batch_size, embed_size, -1).contiguous() ## # (bz, c=128, h*w)
    # Get attention   
    attn = torch.matmul(query, context) / np.sqrt(embed_size) # (bz, rgn_num, h*w)
    attn = F.softmax(attn*smooth, dim=-1)                
#     attn = torch.sigmoid(attn)
#     import pdb
#     pdb.set_trace()
    weightedContext = torch.matmul(attn, torch.transpose(context, 1, 2))
    
    return weightedContext, attn

class LearnedPositionEncoding(nn.Embedding):
    def __init__(self,d_model, dropout = 0.1,max_len = 128*128):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x.view(x.size(0), -1).unsqueeze(1)
        x = x + weight[:x.size(0),:]
        return self.dropout(x)
    
def positional_encoding_1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

    
class View(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

    
class CrossmodalFusion_sp(nn.Module):
    def __init__(self, img_dim, embed_size, raw_feature_norm="clipped_l2norm", lambda_softmax=4.):
        super(CrossmodalFusion_sp, self).__init__()
        self.raw_feature_norm = raw_feature_norm
        self.lambda_softmax = lambda_softmax
        
        self.MLP_image = MLP(embed_size, embed_size // 2, 128, 2) 
            
        self.embedding_Seg = nn.Sequential(
            nn.Conv2d(9, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        self.embed_fusion = MLP(128, embed_size//2, embed_size, 2) 
        
    def refine(self, seg_context, rgn, region_lens):
        
        ## mask
        pe_lens = region_lens.unsqueeze(1).repeat(1, rgn.size(1)).to(region_lens.device)
        mask = torch.arange(rgn.size(1)).expand(rgn.size(0), rgn.size(1)).to(region_lens.device)
        mask = (mask < pe_lens.long())
        pe_lens = pe_lens.masked_fill(mask == 0, 0)   
        fusion_features = self.embed_fusion(seg_context*((mask*1).unsqueeze(-1)) + rgn)
        return fusion_features
    
    def forward(self, rgns, seg_results_key, region_lens):
        '''
        seg_results_key [bt, 128, 128]
        '''
        ##position embedding
        bt, h, w = seg_results_key.shape
        pos_emd = positional_encoding_1d(8, 64*64)
        
        rgns = self.MLP_image(rgns) # [bt, rgn_num, img_dim]-->[bt, rgn_num, embed_size]
        
        seg_results_key = seg_results_key.view(rgns.size(0), -1) # bt 
        seg_results_key = torch.cat((seg_results_key.unsqueeze(-1), pos_emd.repeat(rgns.size(0), 1, 1).to(rgns.device)), dim=-1)
        seg_results_key = seg_results_key.view(bt, h, w,-1).permute(0,3,1,2)
        
        
        seg_results_key = self.embedding_Seg(seg_results_key) # [bt,c=128, h, w]

        ## context is the attented seg results guided by region features
#         context, _ = func_attention(rgns, seg_results_key, self.raw_feature_norm, smooth=self.lambda_softmax)
        context, _ = func_attention(rgns, seg_results_key, self.raw_feature_norm, smooth=self.lambda_softmax)
        # weirgns = [bt, d]  seg_results_Query [bt, d] 
        ref_img = self.refine(context, rgns, region_lens)
        
        del rgns
        return ref_img
    
  
    
    

