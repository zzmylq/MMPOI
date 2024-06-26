# MMPOI

This is the PyTorch implementation of the paper "MMPOI: A Multi-Modal Content-Aware Framework for POI Recommendations"

![model-structure](figures/framework_.png)


## Train

- Download dataset from https://drive.google.com/file/d/17FbNvkO74xub6AeT2fpm938qDqUDB-04/view?usp=sharing.

- Unzip `NYC.zip` to `dataset/`.

- Run `build_graph.py` to construct the sequence graphs from the training data.

- Train the model using `python train.py`. All hyper-parameters are defined in `param_parser.py`

## Raw Dataset 

NYC: https://drive.google.com/file/d/1v0BvKs46ixUf1CgjRlk9MY5JsFL7ILxC/view?usp=sharing![image](https://github.com/zzmylq/MMPOI/assets/49523147/e794ebf0-9e9d-46c8-aef8-1b47333ba5de)

TKY: https://drive.google.com/file/d/1Cpnp2iEmHGfvUkOL-8myeQmyOuq4LmV7/view?usp=sharing![image](https://github.com/zzmylq/MMPOI/assets/49523147/5d8fcdfb-1913-4813-81cc-499b54c3c2c4)

Yelp: https://www.yelp.com/dataset

Please cite our paper if you use this dataset, thank you very much!

## Citation

```
@inproceedings{DBLP:conf/www/XuCZC24,
  author       = {Yang Xu and
                  Gao Cong and
                  Lei Zhu and
                  Lizhen Cui},
  title        = {{MMPOI:} {A} Multi-Modal Content-Aware Framework for {POI} Recommendations},
  booktitle    = {Proceedings of the {ACM} on Web Conference 2024, {WWW} 2024, Singapore,
                  May 13-17, 2024},
  pages        = {3454--3463},
  publisher    = {{ACM}},
  year         = {2024},
  url          = {https://doi.org/10.1145/3589334.3645449},
  doi          = {10.1145/3589334.3645449}
}

```

