# 224UProject
Project for Natural Language Understanding

* Maintains all `.conf` files in the root project directory
* To run the project, invoke "python test/run.py --conf_file [CONF FILENAME]"  
* Note, the format of the `.conf` files is as follows:
    * `model:` specifies the learning algorithm to use
        * choose from among: `log_reg`, `svm`
    * `features:` specifies the features to extract from the data 
        * choose from among: `word_overlap`, `word_cross_product`, `synset`, `hypernym`, `antonym`
    * `feature_file:` specifies the name of the feature file to be generated
    * `load_vectors:` [Optional] indicates whether to use featurized vectors and classification labels already stored on disk in          the given filename.   

