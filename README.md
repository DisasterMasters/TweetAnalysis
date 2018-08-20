<h1>Repo for analyzing tweets from the TweetScraper repo</h1>
<h3>The DATA directory</h3>
<p>This directory contains <i>florida_data</i>, which contains all raw data collected from the TweetScraper. This directory is divided into subdirectories according to whether they are from government, media, utility, or nonprofit source.</p>
<h3>The FLORIDA directory</h3>
<p>The top level of this directory contains all of the code that is actively being used to do machine learning and visualization. <b>The programs must be run separately for every <i>tweet source</i> (gov, media, utility, or nonprofit), since we are running separate analysis on each source.</b></p>
<ul><li><b>crisislex.py</b>: this is a class containing all of the words found in the crisislex. It is imported in <i>lex_dates.py</i></li>
<li><b>doc2vec.py</b>: this program builds the model to use when predicting on categories in predictdoc2vec.py. The model is stored in the doc2vecmodels directory. It is loaded in predictdoc2vec.py. This is one of the supervised training methods used in predicting the categories based on the manual coding. The manually coded tweets are loaded into this program to train the model on, and are located in <i>training_data/supervised_data/&lt;tweetsource&gt;</i><p>USAGE: python doc2vec.py &lt;tweetsource&gt;</p></li>
<li><b>lex_dates.py</b>: this program is for filtering the raw data from the DATA directory and putting into a tab separated and unique delimiter separated format. This program filters data based on whether the tweet contains a word in the crisislex, cleans any dirty text, and only retains tweets made in September. It produces a file in the <i>training_data</i> directory named &lt;tweetsource&gt;_data.txt. which is used in all supervised and unsupervised learning programs, as well as the vizualization programs. <p>USAGE: python lex_dates.py &lt;tweetsource&gt;</p></li>
<li><b>nmf.py</b>: this program is part of the unsupervised learning portion of the project. It will create topics based on the trends it identifies from the data. Currently the number is set to 5.  It produces two files. The first is the topics file, which is stored in <i>results/&lt;tweetsource&gt;_topics.txt</i>. The second is the weights column file, stored in <i>results/W_indices_&lt;tweetsource&gt;</i>. It is a file containing the weights related to each tweet from the tf-idf matrix. A more thorough explanation about what the weights column is is in <i>represent.py</i>.<p>USAGE: python nmf.py &lt;tweetsource&gt;</p></li>
<li><b>predictdoc2vec.py</b>: this program loads the model created in <i>doc2vec.py</i> and predicts categories on the unseen data contained in the DATA directory. It creates the file <i>results/&lt;tweetsource&gt;_supervised_doc2vec.csv</i>, which is later scp'd to the EECS website in my home directory as well as used in the <i>sup_graphs.py</i> program. It contains the tweets and their predicted categories, as well as their dates and permalinks.</b><p>USAGE: python predictdoc2vec.py &lt;tweetsource&gt;</p></li>
<li><b>randomforest.py</b>: this program is part of the supervised learning methods in this directory, and the one used most often. It will build a tf-idf matrix from the manually coded tweets and theirlabels, fit it to a random forest classifier, and then make predictions on the unseen data. It produces the file <i>results/&lt;typeoffile&gt;_supervised_rf.csv</i>, which is later scp'd to the EECS website in my home directory as well as used in the <i>sup_graphs.py</i> program. It contains the tweets and their predicted categories, as well as their dates and permalinks.</b><p>USAGE: python randomforest.py &lt;tweetsource&gt;</p></li>
<li><b>represent.py</b>: this program is used in conjunction with <i>nmf.py</i> in order to find the relevant tweets that correspond to the topics generated from NMF. An explanation of how the program works can be found in the header of the file. It produces the file <i>results/unsupervised_&lt;typeoffile&gt;_tweets.csv</i>, which is later scp'd to the EECS website in my home directory as well as used in the <i>un_graphs.py</i> program. It contains the 5 topics generated, with the tweets and their categories underneath, as well as their dates and permalinks.</b><p>USAGE: python represent.py &lt;tweetsource&gt;</p></li>
<li><b>sup_graphs.py</b>: this program is used to create the graphs that come from the supervised training. The second command line argument will either be n or p. Putting n will produce a graph showing frequency count, while putting p will produce a graph showing percentage. The type of visualization will be reflected in the name of the file. The third command line argument is optional, and should only be included if visualizing results from doc2vec. The program loads the predictions from doc2vec or randomforest and generates an HTML file under <i>results/&lt;tweetsource&gt;_&lt;typeofvisualization&gt;_&lt;supervisedmethod&gt;.html</i>containing an interactive graph. The HTML file is later scp'd to the EECS webhome directory.<p> USAGE: python sup_graphs.py &lt;tweetsource&gt; &lt;n/p&gt; [doc2vec]</p></li>
<li><b>test_doc2vec.py</b>: this program tests the accuracy of the doc2vec supervised learning method on the manually coded tweets. The program will output a percentage correct to the command line. <p>USAGE: python test_doc2vec.py &lt;tweetsource&gt;</p></li> 
<li><b>test_randomforest.py</b>: this program tests the accuracy of the random forest supervised learning method on the manually coded tweets. The program will output a correctness score to the command line (the score is the percentage correct if multiplied by 100). <p>USAGE: python test_randomforest.py &lt;tweetsource&gt;</p></li>
<li><b>un_graphs.py</b>: this program is used to create the graphs that come from the unsupervised training (NMF). The second command line argument will either be n or p. Putting n will produce a graph showing frequency count, while putting p will produce a graph showing percentage. The type of visualization will be reflected in the name of the file. The program loads the predictions from NMF and generates an HTML file under <i>results/&lt;tweetsource&gt;_nmf.html</i>containing an interactive graph. The HTML file is later scp'd to the EECS webhome directory.<p> USAGE: python un_graphs.py &lt;tweetsource&gt; &lt;n/p&gt;</p></li></ul>
<h3>florida/doc2vecmodels directory</h3>
<p>Stores the models generated from doc2vec.py</p>
<h3>florida/results directory</h3>
<p>Stores the results from the unsupervised/supervised methods, the graphs from these methods, and the indices from the tf-idf matrix from NMF</p>
<h3>florida/tmp</h3>
<p>Random programs that might be useful later on. Contains programs to randomize output files for manual coding and some smaller files to use to test code</p>
<h3>florida/training_data</h3>
<p>Contains the files used for supervised/unsupervised predicting, generated from lex_dates.py. These files are what we are predicting categories on. Additionally, contains the Crisislex original txt file.</p>
<h3>florida/training_data/supervised_data</h3>
<p>Contains the manually classified files sent from Xiaojing. Make sure the tweets are in the first column and the manual code is in the second column! (Might have to move columns in file around or adapt code to change depending on format of file sent). The manually coded files are separated into subdirectories depending on their source.</p>
<h3>florida/useless</h3>
<p>Some useless files that I just didn't have the heart to -rm... They are not being used right now but I kept them around just in case.</p>
<h3>florida/webhome</h3>
<p>Contains the index.html file I use on my EECS web directory.</h3>

<h1>Recommendations/Workflow</h1>
<p><i>Right now I am manually running these files, but I would suggest writing a shell script to automate the command line arguments being typed.</i></p>
<p><i>With Puerto Rico coming soon, I would recommend redoing the directory structure in a way that makes more sense and is more efficient to reach both Florida and Puerto Rico tweets.</i></p>
<p>When receiving new data, put it into the DATA directory under its appropriate source.</br>
<br>Then, run lex_dates.py on the source you want to analyze.</br>
<br>After running lex_dates.py, select which method you want (supervised/unsupervised), and run it.</br>
<br>If you chose the unsupervised method, you will have to run represent.py</br>
<br>Then, depending on if you chose unsupervised or supervised, run either sup_graphs.py or un_graphs.py and get your visualizations.</br>





