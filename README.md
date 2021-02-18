# kaggle-rainforest
<div align="center"><img src="./img/001.png"></div>
https://www.kaggle.com/c/rfcx-species-audio-detection/overview  


### Note for my GitHub repository
My repository is separated each branch, and main-branch is not up to date.
- stage1 (stage1 which are little leaky)
- stage1_v2 (stage1 which was fixed cross validation scheme to remove leakage)
- stage2 (stage2 which are little leaky)
- stage2_v2 (stage2 which was fixed cross validation scheme to remove leakage)
- stage3 (stage3 which are little leaky)
- stage3_v2 (stage3 which was fixed cross validation scheme to remove leakage)

### result
5th/1154
<div align="center"><img src="./img/002.png"></div>

  
### log for competition(Notion link)
https://www.notion.so/3b0fab7ffef64e32b83551e32819c353


### Solution
My best model is 3 step

### stage1

CV:0.81 LB:0.84(0123_181828)

- EfficientNet-b2
- SED model(use clipwise output)
- StratifiedKFold 5fold
- 30 epoch
- only TP data
- melspec(244, 400)
- LSEPLoss()
- no augmentation

The purpose of stage1 is to make pretrained model.  

---

### stage2 (not use)
  
CV:  0.734 LB:0.896(0201_213254)
  
- stage1のpretrained model
- 5epoch
- TP/FP all data
- 2 loss(positive/negative)
- FocalLoss(σ=2, α=1)
- batch size16 (TP:FP=1:1 by using batch sampler)
- Sampler
  
The purpose of stage2 is to improve model and make pseudo label by this model.
Now I use toda's pseudo label in stage3. So I don't use this stage2.
Pseudo label was made by stage2 model with threshold=0.5.
  
---
  
### stage3
  
CV:0.954 / LB:0.950 (recall:0.966 prcision:0.558) (0206_232645)
  
- pretrained stage1 model(EfficienetNet-b2)
- 5epoch(10epoch is not good)
- use [toda's pseudo label](https://drive.google.com/file/d/1gSjK4Oq0iPDgy47nbXiIM7ow-b0BowgR/view?usp=sharing)
- Focalloss (posi_loss, nega_loss, zero_loss /detail is later)
- posi_loss weight=2 & zero label smoothing=0.45(+0.009)
- batch size16 (TP:FP=1:1 by using batch sampler)
- last layer mixup (+0.007)
  
Using Focal loss is here.
posi loss is using positive(TP) label and pseudo positive label.
nega loss is using negative(FP) label and pseudo negative label.
zero loss is using other label.(It is ambiguous Labels)

### ensemble
public LB:0.963/Private LB: 0.968  
ensemble model(Resnet128, Efficientnet-b2, ResNeSt50, ViT, WaveNet)