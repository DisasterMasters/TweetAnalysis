<h3>Repository for the code used to analyze tweets from the TweetScraper repo</h3>
<h3>The DATA directory</h3>
<p>This directory contains <i>florida_data</i>, which contains all raw data collected from the TweetScraper. This directory is divided into subdirectories according to whether they are from government, media, utility, or nonprofit source.</p>
<h3>The FLORIDA directory</h3>
<p>The top level of this directory contains all of the code that is actively being used to do machine learning.</p>
<ul><li><b>crisislex.py</b>: this is a class containing all of the words found in the crisislex. It is imported in lex\_dates.py</li>
<li><b>doc2vec.py</b>: this program builds the model to use when predicting on categories in predictdoc2vec.py. The model is stored in the doc2vecmodels directory. It is loaded in predictdoc2vec.py. This is one of the supervised training methods used in predicting the categories based on the manual coding. <p>USAGE: python doc2vec.py &lt;tweetsource&gt; (tweetsource = either utility, nonprofit, gov, or media)</p></li>
<li><b>lex_dates.py</b>: this program is for filtering the raw data from the DATA directory and putting into a tab separated and unique delimiter separated format. This program filters data based on whether the tweet contains a word in the crisislex, and cleans any dirty text, and only retains tweets made in September. It produces a file in the <i>training_data</i> directory named &lt;tweetsource&gt; (media, utility, nonprofit, or gov) \_data.txt. <p>USAGE: python lex\_dates.py &lt;tweetsource&gt;</p>
