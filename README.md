Sentence-Level Entailment Challenge
---------------------------------------

The codebase implements an sklearn pipeline to train and cross-validate a
machine learning classifier on a three-class  pairwise sentence entailment
task.  Data resides in the nli-data/ directory.   

* To run the codebase, invoke "python test/run.py --conf [CONF FILENAME]"
* Maintains all `.conf` files in the root project directory

* The format of the `.conf` files is as follows:
    * `model:` specifies the learning model to use.  
    
	       - Naive Bayes:		    `naive-bayes`
	       - Logistic Regression: 	    `log_reg`
	       - Support Vector Machine:    `svm`
	       - Random Forest: 	    `forest`
	       - Extra Tree Random Forest:  `extra_tree`

    * `features:` specifies the features to extract from the data.  The current list of features templates can
         be seen by running the script `python test/list_features.py`.    
    * `feature_file:` specifies the name of the feature file to be generated. The feature vectors and labels will be stored as
      [feature_file].[`train` | `dev`].[`features` | `labels`].
    * `load_vectors:` [Optional] Indicates whether to use featurized
         vectors and classification labels already stored on disk in
         the given filename.  If load_vectors is set to True, then feature
         vectors will be loaded from the [feature_file] name in output/,
         if it exists.  Setting load_vectors to false is best for
         modifying an existing feature.  
    * `plot` [Optional] : If set to `true`, the classifier will train on a 2-D
         projection of the selected feature set, and the output will be a
         plot of the decision boundary learned, saved in output/plot.png.
    * `param_grid` [Optional]: Gives a range of hyperparameters for the pipeline to optimize over.  If not given, the default 
         optimization grid at the top of features/features.py for the given model will be used.  The expected argument is a 
         python dictionary, with iterables as values (i.e. numpy.arange may be used)

Feature sets are saved by default in output/.  

* ablation.py, found in the test/ folder, builds an ablation study out of
  the configuration file used to run.  It tries each feature 
  with the word_overlap baseline in parallel, and saves in
  "output/wo+[feature_name]".  Arguments are

	* --conf [CONF_FILE_NAME]
	* --mp 	[Number of processes to deploy in parallel]
 
  Note that ANSI color escape does not work when writing to files.  It is
  recommended that you modify the prettyPrint function in util/colors.py
  to exclude color escaping before running ablation.
