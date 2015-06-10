Sentence-Level Entailment Challenge
---------------------------------------

The codebase implements an sklearn pipeline to train and cross-validate a
machine learning classifier on a three-class  pairwise sentence entailment
task.  Data resides in the nli-data/ directory.   

Preparing the Codebase
---------------------
* First, you'll want to download a copy of GigaWord 5B GloVe vectors and save it into the nli-data/ directory.
  The 50-dimensional GLoVe vectors used can be found at http://nlp.stanford.edu/projects/glove/.  
* FrameNet feature parses of the data are already provided.  If you want to use a different data set, you must use parse it
  with the SEMAFOR parser included.
* Ensure that you have nltk installed, along with the WordNet corpus, Lemmatizers, and Taggers.  
* Invocation: "python test/run.py --conf [CONF FILENAME]"
* All `.conf` files are the root project directory

Configuration File Guide
--------------------------
There are six flags in a configuration file `.conf`, which can be specified in any order. 
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
         python dictionary, with iterables as values (i.e. numpy.arange may be used). 
       	 If only one combination of parameter options is given, then the model will fit the singular parameter set, and skip		 cross-validation.  This greatly increases the speed of obtaining results. 

Feature sets are saved by default in output/.  

Testing Tools
-----------------------
* ablation.py, found in the test/ folder, builds an ablation study out of
  the configuration file used to run.  It tries each feature 
  with the word_overlap baseline in parallel, and saves in
  "output/wo+[feature_name]".  Arguments are

	* --conf [CONF_FILE_NAME]
	* --mp 	[Number of processes to deploy in parallel]
 
  Note that ANSI color escape does not work when writing to files.  It is
  recommended that you modify the prettyPrint function in util/colors.py
  to exclude color escaping before running ablation.
* Running with `plot : True` in your `.conf` file will output the decision boundary, as described above.

Known Platform / Version Dependencies:
--------------------------------------
* sklearn > 0.16.  This allows LogisticRegression() to be used with the L-BFGS solver, which is more precise than the standard   linear solver.  If running below sklearn 0.16, recursive feature selection will not work.  So just install 0.16 :)
* OS X only: The default OS X Accelerate Framework does not support calls to Grand Central Dispatch (GCD) on both sides of a    fork(), so Python's multiprocessing implementation, used to speed up grid search and cross-validation, will crash.  Two       possible workarounds:
	- Install numpy and scipy linked against ATLAS / OpenBLAS / MKL, which do not have this programmatic restriction.
	- In params_grid, specify the argument `clf__n_jobs` : [1] if the classifier can be parallelized, and ensure that the           grid you pass in encompasses a singular set of parameter values. 
* Recommended minimum system specs: Intel i5 or higher, 4+ cores, > 2GB RAM free.  
