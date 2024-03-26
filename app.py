import os
import time
import pickle
import re

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.pipeline import Pipeline
import nltk

import base64
from pathlib import Path


#For Project 3
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
bin_file = os.path.join(__location__, "top_banner.jpg")


tokenized_df_path = os.path.join(__location__, "final_tokenized_df.pkl")
model_filepath = os.path.join(__location__, "model.pkl")

# load the train model
with open(model_filepath, 'rb') as handle:
    model = pickle.load(handle)


with open(tokenized_df_path, 'rb') as new_handle:
    tokenized_df = pickle.load(new_handle)


def main():
    st.set_page_config(layout="wide")
    style = """<p style='color:white'>Developed by: Eugene Matthew Cheong | Pius Yee | Conrad Aw </p>
       """
    # Add a banner image
    st.image("top_banner.png", use_column_width=True)  # Adjust the path to your image file
    #st.markdown(img_to_html(bin_file), unsafe_allow_html=True)
    st.markdown(style, unsafe_allow_html=True)
    
    #Setting up Side Bar 
    sidebar_num_post_value = st.sidebar.number_input('Number of Post', step=1, format='%d', value=400)
    sidebar_num_comments_value = st.sidebar.number_input('Number of Comments', step=1, format='%d', value=1)
    sidebar_num_toptrendingwords = st.sidebar.slider('Number of Top Trending Words', step =1,format="%f",min_value=1, max_value=200, value=120)
    sidebar_filter_word_input = st.sidebar.text_input("Filter: (Use commas as separator for mulitple keywords to filter. No spaces.)")
    #Setting up Main
    reddit_input = st.text_input("Input subreddit r/: (Use commas as separator for mulitple reddits) | NOTE: Only r/Parenting is being used for testing.", value="Parenting")
    submit_button = st.button('Submit')
    #Create Columns
    left, right = st.columns((4,1))
    # if button is pressed
    if submit_button:
        reddit_input_list = str(reddit_input).split(",")
        st.success(start_extraction(left,right,sidebar_num_post_value,sidebar_num_comments_value,sidebar_num_toptrendingwords,sidebar_filter_word_input,reddit_input_list))



def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid'> ".format(
      img_to_bytes(img_path)
    )
    return img_html

def gather_comments(comments, comment_dict):
    for comment in comments:
        time.sleep(0.1)
        comment_dict.append(comment.body)
        if len(comment.replies) > 0:
            comment_dict.append(comment.replies)
            gather_comments(comment.replies, comment_dict)
            

def remove_characters(text):
  pattern = '<[^>]*>'
  replacement = ''
  result_string = re.sub(pattern, replacement, str(text))
  return result_string


# def access_reddit_api():
#     # Initialize a Reddit instance with your API credentials
#     reddit = praw.Reddit(
#         client_id='YheGGNwn1zlIePJLrJZZYw',
#         client_secret='JWe8I5cM8YCZGowmL_WPe1d-UuXuFw',
#         user_agent='eumattbro'
#     )

#     return reddit


# def web_scraping(sub_reddits):
        # ext = []
        # reddit = access_reddit_api

        # for sub_reddit in sub_reddits:
        #     for submission in reddit.subreddit(sub_reddit).hot(limit=post_limits):
        #         post_data = ({
        #             "subreddit": sub_reddit,
        #             "title": submission.title,
        #             "selftext": submission.selftext,
        #             "score": submission.score,
        #             "url": submission.url,
        #             })


        #         submission.comments.replace_more(limit=comment_limits)
        #         post_data['comments'] = []

        #         gather_comments(submission.comments, post_data['comments']) 


        #         ext.append(post_data)
        # # Create pandas dataframe
        # reddit_df = pd.DataFrame(ext)

        # return reddit_df


# def clean_scrape_data(reddit_df):
#     # create a new data frame and keep messages from title, selftext and comments
#     final_df = pd.DataFrame({'category':[],'text':[], "mum":[]})
#     index_count = 0
#     for i in range(reddit_df.shape[0]):
#         final_df.loc[index_count] = ["title",reddit_df.loc[i]['title'],reddit_df.loc[i]['subreddit']]
#         index_count += 1
#         final_df.loc[index_count] = ["selftext",reddit_df.loc[i]['selftext'],reddit_df.loc[i]['subreddit']]
#         index_count += 1
#         for cmt in reddit_df.comments: # split different comments into separate rows
#             final_df.loc[index_count] = ["comment",cmt,reddit_df.loc[i]['subreddit']]
#             index_count += 1
    

#     final_df['text'] = final_df['text'].apply(remove_characters)

#     # further remove comments with "[removed] and [deleted]"
#     final_df = final_df[~final_df['text'].apply(lambda x: any(word in x for word in ['[removed]','[deleted]']))]
    
#     final_df['text'] = final_df['text'].replace('\n', '')

#     # Removing special characters from the 'text' column
#     final_df['text'] = final_df['text'].replace(r'[^a-zA-Z0-9\?\! ]', '', regex=True)

#     # To identify the auto message, we find similar message which more than 10 letters
#     # create a new data frame
#     check_auto = pd.DataFrame(final_df.text.value_counts())
#     check_auto = check_auto.reset_index()
#     # identify auto bot message with duplicate same message and more than 20 words
#     check_auto = check_auto[(check_auto.text.str.len() >20) & (check_auto['count'] > 2)]

#     # remove the bot message from the dataframe
#     final_df = final_df[~final_df['text'].isin(list(check_auto.text))]

#     final_df = final_df[~final_df['text'].map(lambda x: x == "" or pd.isnull(x))]

#     return final_df


# def tokenize_data(df):
#     # instantiate Tokenizer with Regex
#     tokenizer = RegexpTokenizer(r'[^\d\W]+') # keep words only  

#     # "Run" Tokenizer and create new column for clean data
#     df['text'] = df['text'].astype("str")
#     df['text'] = [tokenizer.tokenize(x.lower()) for x in list(df.text)]

#     # Remove stopwords from "spam_tokens."
#     df['text'] = df['text'].apply(lambda x: [token for token in x if token not in stopwords.words('english')])

#     # Instantiate lemmatizer.
#     lemmatizer = WordNetLemmatizer()
#     lst = []
#     for row in list(df['text']):
#         lst.append([lemmatizer.lemmatize(i) for i in row])

#     df['text'] = lst
#     df['text'] = df['text'].astype("string")

#     text_result = df.text

#     return text_result

def start_extraction(left,right,post_limits,comment_limits,num_of_trending_words,filter_words,sub_reddits):
    with st.spinner("Extracting"):
        time.sleep(2)

        result_X = tokenized_df.text
        tokenized_df["mum"] = model.predict(result_X)
    
    st.success("Finish Extraction")
    with st.spinner("Processing"):

        mum_found_list = []

        for p in list(tokenized_df[tokenized_df.mum == 1].text):
            p = p.replace('[', '').replace(']', '').replace("'", "")
            currentword_list = p.split(', ')
            for pp in currentword_list: 
                mum_found_list.append(pp)
        

        try:
            filter_words_list = filter_words.split(",")
        except:
            filter_words_list = list(filter_words)
            
        for filter_word in filter_words_list:
            for index, mum_word in enumerate(mum_found_list):
                if mum_word == filter_word:
                    del mum_found_list[index]

        mum_found_result = ', '.join(mum_found_list)

        
    st.success("Done!")

    try:
        show_word_cloud(left,num_of_trending_words,mum_found_result)
        show_ngram(mum_found_list)
    except:
        st.error("There are no words available to generate a Word Cloud.")



def show_word_cloud(column,num_of_words,text):
    # Create and generate a word cloud image:
    wordcloud = WordCloud(max_words=num_of_words).generate(text)

    # Display the generated image:
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    plt.gcf().set_facecolor('none')
    plt.rcParams['text.color'] = 'white'
    column.pyplot(fig)


def show_ngram(text_list):
# plotting for N-gram
    fig, axes = plt.subplots(2, 2, figsize=(15,50)) 
    pd.Series(nltk.ngrams(text_list, 1)).value_counts().sort_values()[-20:].plot(kind='barh',title='Trending Words (1-gram)',ax=axes[0,0], xlabel='count')
    pd.Series(nltk.ngrams(text_list, 2)).value_counts().sort_values()[-20:].plot(kind='barh',title='Trending Words (2-gram)',ax=axes[0,1], xlabel='count')
    pd.Series(nltk.ngrams(text_list, 3)).value_counts().sort_values()[-20:].plot(kind='barh',title='Trending Words (3-gram)',ax=axes[1,0], xlabel='count')
    pd.Series(nltk.ngrams(text_list, 4)).value_counts().sort_values()[-20:].plot(kind='barh',title='Trending Words (4-gram)',ax=axes[1,1], xlabel='count')

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.6, wspace=0.5, hspace=0.2)
    
    plt.gcf().set_facecolor('none')

    params = {"ytick.color" : "white",
            "xtick.color" : "white",
            "axes.labelcolor" : "white",
            "axes.edgecolor" : "white"}
    plt.rcParams.update(params)
    st.pyplot(fig)



if __name__ == '__main__':
    main()



