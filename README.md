# 224UProject
Project for Natural Language Understanding

* Maintains all `.conf` files in the root project directory
* To run the project, invoke "python test/run.py --conf [CONF FILENAME]"  

* The format of the `.conf` files is as follows:
    * `model:` specifies the learning model to use.
        * choose from among: `log_reg`, `svm`, `naive_bayes`
    * `features:` specifies the features to extract from the data.  The current list of features templates can
         be seen by running the script `python test/list_features.py`.    
    * `feature_file:` specifies the name of the feature file to be generated. The feature vectors and labels will be stored as
      [feature_file].[`train` | `dev`].[`features` | `labels`].
    * `load_vectors:` [Optional] Indicates whether to use featurized vectors and classification labels already stored on disk in          the given filename.   If `load_vectors` is true, then all featurized vectors and labels for train and dev are expected
         to exist.  If a previous job was aborted before saving the dev feature vectors, `load_vectors` should be set to False. 
    * `param_grid:` [Optional] Gives a range of hyperparameters for the pipeline to optimize over.  If not given, the default 
         optimization grid at the top of features/features.py for the given model will be used.  The expected argument is a 
         python dictionary, with iterables as values (i.e. numpy.arange may be used)

