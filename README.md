# Aspect-level Sentiment Classification
Code and dataset for ACL2018 [[paper]](https://arxiv.org/abs/1806.04346) ‘‘Exploiting Document Knowledge for Aspect-level Sentiment Classification’’. 

## Data
The preprocessed aspect-level datasets can be downloaded at [[Download]](https://drive.google.com/open?id=1e7WnaWWDggJ8r8t3FhS4RSp88Ev2gXYs), and the document-level datasets can be downloaded at [[Download]](https://drive.google.com/open?id=15NTGPsO_E8dwti_QILNMwLQnUaTn3auq). The zip files should be decompressed and put in the main folder.

The pre-trained Glove vectors (on 840B tokens) are used for initializing word embeddings. You can download the extracted subset of Glove vectors for each dataset at [[Download]](https://drive.google.com/open?id=1f8OlsHv82tIX0CyxbUqywew08StUjVLp), the size of which is much smaller. The zip file should be decompressed and put in the main folder.

## Training and evaluation

### Pretraining on document-level dataset
The pretrained weights from document-level examples used in our experiments are provided at pretrained_weights/. You can use them directly for initialising aspect-level models.

Or if you want to retrain on ducment-level again, execute the command below under code_pretrain/:
```
CUDA_VISIBLE_DEVICES="0" python pre_train.py \
--domain $domain \
```
where *$domain* in ['yelp_large', 'electronics_large'] denotes the corresponding document-level domain. The trained model parameters will be saved under pretrained_weights/. You can find more arguments defined in pre_train.py with default values used in our experiments.

### Training and evaluation on aspect-level dataset
To train aspect-level sentiment classifier, excute the command below under code/:
```
CUDA_VISIBLE_DEVICES="0" python train.py \
--domain $domain \
--alpha 0.1 \
--is-pretrain 1 \
```
where *$domain* in ['res', 'lt', 'res_15', 'res_16'] denotes the corresponding aspect-level domain. *--alpha* denotes the weight of the document-level training objective (\lamda in the paper). *--is-pretrain* is set to either 0 or 1, denoting whether to use pretrained weights from document-level examples for initialisition. You can find more arguments defined in train.py with default values used in our experiments. At the end of each epoch, results on training, validation and test sets will be printed respectively.

## Dependencies
* Python 2.7
* Keras 2.1.2
* tensorflow 1.4.1
* numpy 1.13.3

## Cite
If you use the code, please cite the following paper:
```
@InProceedings{he-EtAl:2018,
  author    = {He, Ruidan  and  Lee, Wee Sun  and  Ng, Hwee Tou  and  Dahlmeier, Daniel},
  title     = {Exploiting Document Knowledge for Aspect-level Sentiment Classification},
  booktitle = {Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics},
  publisher = {Association for Computational Linguistics}
}
```



