# HCLNet — reference implementation on FDCompCN

A runnable implementation of HCLNet ("Hypergraph-based contrastive learning for
enhanced fraud detection", Wang, Shen & Dong, *Frontiers in AI* 8:1703135, 2025).

```
HCLNet/
  data.py      FDCompCN loader
  model.py     MG-HAM, HDV-CL, prediction and loss
  train.py     training / validation / test, checkpoint
  config.yaml  the hyperparameters reported for the YelpChi setting
  checkpoints/hclnet_fdcompcn.pt
  logs/hclnet_fdcompcn.json
```

