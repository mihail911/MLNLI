# 224UProject
Project for Natural Language Understanding

* Maintains all `.conf` files in the root project directory

* Note, the format of the `.conf` files is as follows:
    * `model:` specifies the learning algorithm to use
        * choose from among: `log_reg`, `svm`
    * `features:` specifies the features to extract from the data 
        * choose from among: `word_overlap`, `word_cross_product`, `synset`, `hypernym`, `antonym`
    * `feature_file:` specifies the name of the feature file to be generated

