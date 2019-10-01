#!/usr/bin/env python
# coding: utf-8

# # Phishing Website Detection 

# Step -1 : Data preprocessing 
# 
# This dataset contains few website links (Some of them are legitimate websites and a few are fake websites)
# 
# Pre-Processing the data before building a model and also Extracting the features from the data based on certain conditions

# In[1]:


#importing numpy and pandas which are required for data pre-processing
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets


# In[2]:


#Loading the data
raw_data = pd.read_csv("dataset2.csv") 


# In[3]:


print(raw_data.columns)


# In[4]:


print(raw_data.shape)


# In[5]:


raw_data.head()


# We need to split the data according to parts of the URL
# 
# A typical URL could have the form http://www.example.com/index.html, which indicates a protocol (http), a hostname (www.example.com), and a file name (index.html).

# In[6]:


raw_data['URL'].str.split("://").head() #Here we divided the protocol from the entire URL. but need it to be divided it 
                                                 #seperate column


# In[7]:


seperation_of_protocol = raw_data['URL'].str.split("://",expand = True) #expand argument in the split method will give you a new column


# In[8]:


seperation_of_protocol.head()


# In[9]:


type(seperation_of_protocol)


# In[10]:


seperation_domain_name = seperation_of_protocol[1].str.split("/",1,expand = True) #split(seperator,no of splits according to seperator(delimiter),expand)


# In[11]:


type(seperation_domain_name)


# In[12]:


seperation_domain_name.columns=["domain_name","address"] #renaming columns of data frame


# In[13]:


seperation_domain_name.head()


# In[14]:


#Concatenation of data frames
splitted_data = pd.concat([seperation_of_protocol[0],seperation_domain_name],axis=1)


# In[15]:


splitted_data.columns = ['protocol','domain_name','address']


# In[16]:


splitted_data.head()


# In[17]:


splitted_data['is_phished'] = pd.Series(raw_data['Target'], index=splitted_data.index)


# In[18]:


splitted_data


# Domain name column can be further sub divided into domain_names as well as sub_domain_names 
# 
# Similarly, address column can also be further sub divided into path,query_string,file..................

# In[19]:


type(splitted_data)


# ### Features Extraction

# 
# Feature-1
# 
# 1.Long URL to Hide the Suspicious Part
# 
# If the length of the URL is greater than or equal 54 characters then the URL classified as phishing
# 
# 
# 0 --- indicates legitimate
# 
# 1 --- indicates Phishing
# 
# 2 --- indicates Suspicious

# In[20]:


def long_url(l):
    l= str(l)
    """This function is defined in order to differntiate website based on the length of the URL"""
    if len(l) < 54:
        return 0
    elif len(l) >= 54 and len(l) <= 75:
        return 2
    return 1


# In[21]:


#Applying the above defined function in order to divide the websites into 3 categories
splitted_data['long_url'] = raw_data['URL'].apply(long_url) 


# In[22]:


#Will show the results only the websites which are legitimate according to above condition as 0 is legitimate website
splitted_data[splitted_data.long_url == 0] 


# Feature-2
# 
# 2.URL’s having “@” Symbol
# 
# Using “@” symbol in the URL leads the browser to ignore everything preceding the “@” symbol and the real address often follows the “@” symbol.
# 
# IF {Url Having @ Symbol→ Phishing
#     Otherwise→ Legitimate }
# 
# 
# 0 --- indicates legitimate
# 
# 1 --- indicates Phishing
# 

# In[23]:


def have_at_symbol(l):
    """This function is used to check whether the URL contains @ symbol or not"""
    if "@" in str(l):
        return 1
    return 0
    


# In[24]:


splitted_data['having_@_symbol'] = raw_data['URL'].apply(have_at_symbol)


# In[25]:


splitted_data


# Feature-3
# 
# 3.Redirecting using “//”
# 
# The existence of “//” within the URL path means that the user will be redirected to another website.
# An example of such URL’s is: “http://www.legitimate.com//http://www.phishing.com”. 
# We examine the location where the “//” appears. 
# We find that if the URL starts with “HTTP”, that means the “//” should appear in the sixth position. 
# However, if the URL employs “HTTPS” then the “//” should appear in seventh position.
# 
# IF {ThePosition of the Last Occurrence of "//" in the URL > 7→ Phishing
#     
#     Otherwise→ Legitimate
# 
# 0 --- indicates legitimate
# 
# 1 --- indicates Phishing
# 

# In[26]:


def redirection(l):
    """If the url has symbol(//) after protocol then such URL is to be classified as phishing """
    if "//" in str(l):
        return 1
    return 0


# In[27]:


splitted_data['redirection_//_symbol'] = seperation_of_protocol[1].apply(redirection)


# In[28]:


splitted_data.head()


# Feature-4
# 
# 4.Adding Prefix or Suffix Separated by (-) to the Domain
# 
# The dash symbol is rarely used in legitimate URLs. Phishers tend to add prefixes or suffixes separated by (-) to the domain name
# so that users feel that they are dealing with a legitimate webpage. 
# 
# For example http://www.Confirme-paypal.com/.
#     
# IF {Domain Name Part Includes (−) Symbol → Phishing
#     
#     Otherwise → Legitimate
#     
# 1 --> indicates phishing
# 
# 0 --> indicates legitimate
#     

# In[29]:


def prefix_suffix_seperation(l):
    if '-' in str(l):
        return 1
    return 0


# In[30]:


splitted_data['prefix_suffix_seperation'] = seperation_domain_name['domain_name'].apply(prefix_suffix_seperation)


# In[31]:


splitted_data.head()


# Feature - 5
# 
# 5. Sub-Domain and Multi Sub-Domains
# 
# The legitimate URL link has two dots in the URL since we can ignore typing “www.”. 
# If the number of dots is equal to three then the URL is classified as “Suspicious” since it has one sub-domain.
# However, if the dots are greater than three it is classified as “Phishy” since it will have multiple sub-domains
# 
# 0 --- indicates legitimate
# 
# 1 --- indicates Phishing
# 
# 2 --- indicates Suspicious
# 

# In[32]:


def sub_domains(l):
    l= str(l)
    if l.count('.') < 3:
        return 0
    elif l.count('.') == 3:
        return 2
    return 1


# In[33]:


splitted_data['sub_domains'] = splitted_data['domain_name'].apply(sub_domains)


# In[34]:


splitted_data


# ### Classification of URLs using Random forest 

# In[35]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# In[36]:


#Features
x = splitted_data.columns[4:9]
x   


# In[37]:


#variable to be predicted; yes = 0 and no = 1
y = pd.factorize(splitted_data['is_phished'])[0]
y 


# In[38]:


# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_estimators=100,n_jobs=2,random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(splitted_data[x], y) 


# In[39]:


def predictor(splittted_data):

    print(" script rf_model")
    # load the model from disk
    
    
    print("model loaded")
    print(splittted_data.shape)
    print(list(splittted_data))
   
    preds = clf.predict(splittted_data[x])
    print("prediction complete")
    print(preds)
    if preds == 0:
        str1 = "Spoofed Webpage: YES"
       
    else: str1 = "Spoofed Webpage: NO"
       
    if preds == 0:
        score = score = clf.predict_proba(splittted_data[x])
       
    else: score = 1-(clf.predict_proba(splittted_data[x]))
    
    str2 = "Confidence score: "+ str(score[0][1])

    return str1,str2


# In[40]:


class feature_extractor:

    def __init__(self,url:str):
        self.input_url = url

    def long_url(self,l):
        """This function is defined in order to differntiate website based on the length of the URL"""
        l= str(l)
        if len(l) < 54:
            return 0
        elif len(l) >= 54 and len(l) <= 75:
            return 2
        return 1

    def have_at_symbol(self,l):
        """This function is used to check whether the URL contains @ symbol or not"""
        if "@" in str(l):
            return 1
        return 0

    def redirection(self,l):
        """If the url has symbol(//) after protocol then such URL is to be classified as phishing """
        if "//" in str(l):
            return 1
        return 0

    def prefix_suffix_seperation(self,l):
        """seprate prefix and suffix"""
        if '-' in str(l):
            return 1
        return 0

    def sub_domains(self,l):
        """check the subdomains"""
        l= str(l)
        if l.count('.') < 3:
            return 0
        elif l.count('.') == 3:
            return 2
        return 1


    def extract(self):
        print("in script 2")
        input_data = [{"URL":self.input_url}]
        print('input taken')
        temp_df = pd.DataFrame(input_data)
        print("dataframe created")
        #expand argument in the split method will give you a new column
        seperation_of_protocol = temp_df['URL'].str.split("://",expand = True)
        print("step 1 done")
        #split(seperator,no of splits according to seperator(delimiter),expand)
        seperation_domain_name = seperation_of_protocol[1].str.split("/",1,expand = True)
        print("step 2 done")
        #renaming columns of data frame
        seperation_domain_name.columns=["domain_name","address"]
        print("step 3 done")
        #Concatenation of data frames
        splitted_data = pd.concat([seperation_of_protocol[0],seperation_domain_name],axis=1)
        print("step 4 done")

        splitted_data.columns = ['protocol','domain_name','address']
        print("step 5 done")

        #splitted_data['is_phished'] = pd.Series(temp_df['Target'], index=splitted_data.index)
        #print("step 6 done")

        """feature extraction starts here"""
        #Applying the above defined function in order to divide the websites into 3 categories
        splitted_data['long_url'] = temp_df['URL'].apply(self.long_url)
        print("feature extra 1")
        splitted_data['having_@_symbol'] = temp_df['URL'].apply(self.have_at_symbol)
        print("feature extra 2")
        splitted_data['redirection_//_symbol'] = seperation_of_protocol[1].apply(self.redirection)
        print("feature extra 3")
        splitted_data['prefix_suffix_seperation'] = seperation_domain_name['domain_name'].apply(self.prefix_suffix_seperation)
        print("feature extra 4")
        splitted_data['sub_domains'] = splitted_data['domain_name'].apply(self.sub_domains)
        print("feature extra 5")
        #splitted_data.to_csv(r'dataset3.csv',header= True)

        

        return predictor(splitted_data)


# In[41]:


class Ui_Spam_detector(object):
    def setupUi(self, Spam_detector):
        Spam_detector.setObjectName("Spam_detector")
        Spam_detector.resize(521, 389)
        self.centralwidget = QtWidgets.QWidget(Spam_detector)
        self.centralwidget.setObjectName("centralwidget")

        """check button code and its connectivity to button_click function"""
        self.check_button = QtWidgets.QPushButton(self.centralwidget)
        self.check_button.setGeometry(QtCore.QRect(210, 170, 93, 28))
        self.check_button.setObjectName("check_button")
        self.check_button.clicked.connect(self.button_click)

        """url input section"""
        self.url_input = QtWidgets.QLineEdit(self.centralwidget)
        self.url_input.setGeometry(QtCore.QRect(70, 111, 431, 31))
        self.url_input.setObjectName("url_input")
        
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 110, 81, 31))
        self.label.setObjectName("label")
        
        
        
        """output message"""
        self.output_text = QtWidgets.QTextEdit(self.centralwidget)
        self.output_text.setGeometry(QtCore.QRect(30, 241, 461, 121))
        self.output_text.setObjectName("output_text")
        
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(110, 10, 311, 41))
        self.label_2.setObjectName("label_2")
        
        Spam_detector.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(Spam_detector)
        self.statusbar.setObjectName("statusbar")
        Spam_detector.setStatusBar(self.statusbar)

        self.retranslateUi(Spam_detector)
        QtCore.QMetaObject.connectSlotsByName(Spam_detector)

    def retranslateUi(self, Spam_detector):
        _translate = QtCore.QCoreApplication.translate
        Spam_detector.setWindowTitle(_translate("Spam_detector", "MainWindow"))
        self.check_button.setText(_translate("Spam_detector", "Check "))
        self.label.setText(_translate("Spam_detector", "<html><head/><body><p><span style=\" font-size:10pt;\">URL :</span></p></body></html>"))
        self.label_2.setText(_translate("Spam_detector", "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt;\">Spam URL Detector</span></p></body></html>"))

    def button_click(self):
        text = self.url_input.text()
        #print(text)
        obj = feature_extractor(text)
        str1,str2 = obj.extract()

        self.output_text.append("{} \n{}\n\n".format(str1,str2))
        

    #def show_output():

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Spam_detector = QtWidgets.QMainWindow()
    ui = Ui_Spam_detector()
    ui.setupUi(Spam_detector)
    Spam_detector.show()
    sys.exit(app.exec_())

