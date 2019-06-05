# BlackFridayDataScienceRepo

Black Friday is one of the biggest and busiest days of the year in most parts of the world.
Consumers have been spending millions of dollars on shopping on Black Friday. Although this 
is a big day for retailers, they must plan it accordingly and everything must go right before they
can see big profits and an increase in sales. In this age of technology, retailers can capture
various data about their consumers that can help them in analyzing different factors when it
comes to shopping.

The problem I faced was trying to understand the customer purchase behaviour on different
products. I analyzed consumer behaviour by using linear regression model, random forest classifer model 
and apripori algorithm. Linear regesssion was used to predict purchases against features city, product cateogries 1 & 3. 
Random forest classifier was used to predict the gender of consumers who purchase products and apriori algorithm was used
to find frequent itemsets within the data (an item is a product).
I believe that by analyzing these features it will give the anonymous store onwner insight into sales
analytics and the characteristics that drive the purchasing power of specific products.


First you need to download the “Project Files” folder and save it anywhere on your computer. However, 
it is recommended that you save this folder on your desktop for a simple file path. You do not need to download 
python separately since installing Anaconda means you are getting a version of Python installed.

You can follow the directions of how to install Anaconda Navigator on your operating software here:

For Mac OS http://docs.continuum.io/anaconda/install/mac-os/

For Windows http://docs.continuum.io/anaconda/install/windows/

For Linux http://docs.continuum.io/anaconda/install/linux/


For each link their will be a choice of which version of python to download you may choose which ever version 
of Python you want either 3.7 or 2.7 version. Download the version of python you want because using either version 
will not alter results of our models and algorithms.

After you install Anaconda Navigator you should open the application and then launch the Spyder application 
within the Navigator’s home screen. After you open Spyder, open the files “BlackFridayAnalysis.py” from 
the “ITEC 4230 Project Files” then run your program by copy and pasting code into the IPython console. 
We recommend copying and pasting code into the IPython console in portions because the load time may 
take long if you attempt to run all the code at once. 

The portions you should copy and paste are:
•	Lines 10-30
•	Lines 32-44
•	Lines 48-107
•	Lines 113-132
•	Lines 130-137
•	Lines 144-168
•	Lines 175-211
•	Lines 217-252

Note that if you receive the error:
IOError: File ../User/Desktop/BlackFriday.csv does not exist

Then you need to configure the file path in the variable df to the file path where you downloaded 
the BlackFriday.csv file. For example, Khalid downloaded the file on his MacBook Air laptop so his file path was represented as:  
df = pd.read_csv('../User/Desktop/BlackFriday.csv')

Once you configure the variable df with the correct file path the error should be gone. However, 
if you are still receiving this error then you should close and restart both the Spyder application 
and Anaconda Navigator application.

Also if you receive an error saying "No module named “mlxtend.preprocessing" then write 
this "!pip install mlxtend" in the IPython console to install mlxtend. For similar issues faced with 
other modules replace “mlxtend” with the required module.

