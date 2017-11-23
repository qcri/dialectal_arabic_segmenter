# Dialectal Arabic Segmenter 
Dialectal Arabic Segmenter is a freeware module developed by the ALT team at Qatar Computing Research Institute (QCRI) to process Dialectal Arabic. The segmenter is built using a collection of tweets from frour regions - Egypt, Gulf, Maghrib and Levantine.
 
Arabic Dialects Segmenter implemented using Keras/BiLSTM/ChainCRF. 

# Requirements

This segmenter requires the following packages:

- Python version 2.7
    - Python 3.4 should be fine as well (some minor changes)
    
- `tensorflow` version 0.9 or later: https://www.tensorflow.org
- `keras` version 1.2.2 or later: http://keras.io
- `nltk` version 3.0 or later

## Installation

You can install the Dialectal Arabic Segmenter by cloning the repo:

### Installing Dialectal Arabic Segmenter from github
Clone the repo from the github using the following command:
```
git clone https://github.com/qcri/dialectal_segmenter.git
```
Or download the compressed file of the project, extract it.

## Getting started
Dialectal Arabic Segmenter reads an input Arabic text  file from the stdin and produces the segmentation line per line. The segmenter expects the input file encoded in ``UTF-8``,
```
python code/dialects_segmenter.py -i [in-file] -o [out-file] 
```

For more details see:

``` 
python code/dialects_segmenter.py -h
```


## Publications
Younes Samih, Mohamed Eldesouki, Mohammed Attia, Kareem Darwish, Ahmed Abdelali, Hamdy Mubarak, Laura Kallmeyer, (2017), [Learning from Relatives: Unified Dialectal Arabic Segmentation](http://www.aclweb.org/anthology/K17-1043), Journal Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017), Pages 432-441.

Mohamed Eldesouki, Younes Samih, Ahmed Abdelali, Mohammed Attia, Hamdy Mubarak, Kareem Darwish, Kallmeyer Laura, (2017), [Arabic Multi-Dialect Segmentation: bi-LSTM-CRF vs. SVM](https://arxiv.org/pdf/1708.05891.pdf), arXiv preprint arXiv:1708.05891.

Younes Samih, Mohammed Attia, Mohamed Eldesouki, Ahmed Abdelali, Hamdy Mubarak, Laura Kallmeyer, Kareem Darwish, (2017), [A Neural Architecture for Dialectal Arabic Segmentation](http://www.aclweb.org/anthology/W17-1306), Journal Proceedings of the Third Arabic Natural Language Processing Workshop, Pages 46-54.





## Support

You can ask questions and join the development discussion:

- On the [Dialectal Arabic Tools Google group](https://groups.google.com/forum/#!forum/dat-users).
- On the [Dialectal Arabic Tools Slack channel](https://datsteam.slack.com). Use [this link](https://dat-slack-autojoin.herokuapp.com/) to request an invitation to the channel.

You can also post **bug reports and feature requests** (only) in [Github issues](https://github.com/qcri/dialectal_arabic_tools/issues). Make sure to read [our guidelines](https://github.com/qcri/dialectal_arabic_tools/blob/master/CONTRIBUTING.md) first.


## License

Dialectal Arabic Segmenter is covered by the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).


------------------