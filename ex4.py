import pandas as pd
import sqlite3
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from collections import Counter
import re

def main():
    # Download necessary NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('vader_lexicon', quiet=True)

    sid = SentimentIntensityAnalyzer()

    # Connect to SQLite database and load posts
    conn = sqlite3.connect('database.sqlite')
    posts_data = pd.read_sql_query("SELECT * FROM posts", conn)
    comments_data = pd.read_sql_query("SELECT * FROM comments", conn)
    conn.close()
    
    print(f"Loaded {len(posts_data)} posts and {len(comments_data)} comments for analysis")
    
    # Enhanced stopwords list
    stop_words = set(stopwords.words('english'))
    additional_stopwords = {
        'would', 'could', 'should', 'like', 'get', 'go', 'know', 'think', 'see',
        'one', 'also', 'back', 'even', 'well', 'much', 'really', 'many', 'still',
        'today', 'day', 'time', 'first', 'last', 'next',
        'post', 'posting', 'share', 'shared', 'update', 'news', 'content',
        'people', 'thing', 'something', 'anything', 'someone', 'everyone'
    }
    stop_words.update(additional_stopwords)

    lemmatizer = WordNetLemmatizer()
    
    def preprocess_text(text):
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)     # Remove mentions
        text = re.sub(r'#\w+', '', text)     # Remove hashtags but keep words
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Keep only letters and spaces
        
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        tokens = [t for t in tokens if len(t) > 2 and t.isalpha() and t not in stop_words]
        return tokens
    
    def analyze_sentiment(text):
        sentiment_scores = sid.polarity_scores(text)
        compound = sentiment_scores['compound']
        
        if compound >= 0.05:
            sentiment_class = 'positive'
        elif compound <= -0.05:
            sentiment_class = 'negative'
        else:
            sentiment_class = 'neutral'
            
        return {
            'compound': compound,
            'positive': sentiment_scores['pos'],
            'negative': sentiment_scores['neg'],
            'neutral': sentiment_scores['neu'],
            'classification': sentiment_class
        }

    processed_data = []
    bow_list = []
    
    for _, row in posts_data.iterrows():
        text = row['content']
        tokens = preprocess_text(text)
        sentiment = analyze_sentiment(text)
        
        if len(tokens) > 5:
            processed_data.append({
                'content': text,
                'tokens': tokens,
                'sentiment': sentiment,
                'type': 'post'
            })
            bow_list.append(tokens)

    for _, row in comments_data.iterrows():
        text = row['content']
        tokens = preprocess_text(text)
        sentiment = analyze_sentiment(text)
        
        if len(tokens) > 5:
            processed_data.append({
                'content': text,
                'tokens': tokens,
                'sentiment': sentiment,
                'type': 'comment'
            })
            bow_list.append(tokens)

    # build bigrams
    bigram = Phrases(bow_list, min_count=5, threshold=100)
    bigram_phraser = Phraser(bigram)
    bow_list = [bigram_phraser[doc] for doc in bow_list]

    # trigrams
    trigram = Phrases(bigram[bow_list], min_count=3, threshold=50)
    trigram_phraser = Phraser(trigram)
    bow_list = [trigram_phraser[bigram_phraser[doc]] for doc in bow_list]
    
    # POS(Part-of-Speech) filtering
    def pos_filter(tokens):
        pos_tags = nltk.pos_tag(tokens)
        allowed = {'NN','NNS','NNP','NNPS','JJ','JJR','JJS'}
        return [t for t, tag in pos_tags if tag in allowed]

    bow_list = [pos_filter(doc) for doc in bow_list]

    print(f"After preprocessing, {len(bow_list)} posts remain for analysis")
    
    if len(bow_list) == 0:
        print("No valid posts found for analysis")
        return
    
    # Create dictionary and corpus
    dictionary = Dictionary(bow_list)
    # Filter words that can appear
    dictionary.filter_extremes(no_below=3, no_above=0.35, keep_n=3000)
    corpus = [dictionary.doc2bow(tokens) for tokens in bow_list]
    
    print(f"dictionary size: {len(dictionary)}")

    # optimal_ranking = "looking", "top10" or "decided"
    Optimal_ranking = "decided"
    if Optimal_ranking == "looking":
        # Find optimal number of topics
        optimal_coherence = -100
        optimal_lda = None
        optimal_k = 0
        
        print("\nTesting different numbers of topics...")
        for K in range(9, 15):
            lda = LdaModel(corpus, num_topics=K, id2word=dictionary, passes=15, random_state=42, alpha='auto')
            
            coherence_model = CoherenceModel(model=lda, texts=bow_list, dictionary=dictionary, coherence='c_v')
            coherence_score = coherence_model.get_coherence()
            
            if coherence_score > optimal_coherence:
                print(f"✓ {K} topics: coherence = {coherence_score:.4f} (new best)")
                optimal_coherence = coherence_score
                optimal_lda = lda
                optimal_k = K
            else:
                print(f"  {K} topics: coherence = {coherence_score:.4f}")
        
        print(f"\nOptimal number of topics: {optimal_k}")
        print("Ranking of Topics by Popularity:")
        topic_counts = Counter()
        topic_keywords = {}

        for topic_id in range(optimal_k):
            try:
                words = [w for w, _ in optimal_lda.show_topic(topic_id, topn=5)]
                topic_keywords[topic_id] = ', '.join(words)
            except Exception:
                topic_keywords[topic_id] = "N/A"
        
        for _, bow in enumerate(corpus):
            topic_dist = optimal_lda.get_document_topics(bow)
            if topic_dist:
                dominant_topic = max(topic_dist, key=lambda x: x[1])[0]
                topic_counts[dominant_topic] += 1

        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (topic_id, count) in enumerate(sorted_topics[:10], 1):
            percentage = (count / len(corpus)) * 100
            keywords = topic_keywords.get(topic_id, "N/A")
            print(f"{rank}. Topic {topic_id+1}: {count} posts ({percentage:.1f}%) - Keywords: {keywords}")
    elif Optimal_ranking == "top10":
        # find top 10 topics regardless of coherence score
        lda_top10 = LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15, random_state=42, alpha='auto')
        coherence_model = CoherenceModel(model=lda_top10, texts=bow_list, dictionary=dictionary, coherence='c_v')
        coherence_score_top10 = coherence_model.get_coherence()

        print(f"\ncoherence score of 10 topics = {coherence_score_top10:.4f}")
        print("Ranking of 10 Topics by Popularity Regardless of Coherence Score:")
        topic_counts_top10 = Counter()
        topic_keywords_top10 = {}

        for topic_id in range(lda_top10.num_topics):
            try:
                words = [w for w, _ in lda_top10.show_topic(topic_id, topn=5)]
                topic_keywords_top10[topic_id] = ', '.join(words)
            except Exception:
                topic_keywords_top10[topic_id] = "N/A"

        for _, bow in enumerate(corpus):
            topic_dist = lda_top10.get_document_topics(bow)
            if topic_dist:
                dominant_topic = max(topic_dist, key=lambda x: x[1])[0]
                topic_counts_top10[dominant_topic] += 1

        sorted_topics = sorted(topic_counts_top10.items(), key=lambda x: x[1], reverse=True)

        for rank, (topic_id, count) in enumerate(sorted_topics[:10], 1):
            percentage = (count / len(corpus)) * 100
            keywords = topic_keywords_top10.get(topic_id, "N/A")
            print(f"{rank}. Topic {topic_id+1}: {count} posts ({percentage:.1f}%) - Keywords: {keywords}")
    elif Optimal_ranking == "decided":
        print("\nTraining LDA model with 78 topics...")
        lda_model = LdaModel(
            corpus, 
            num_topics=78, 
            id2word=dictionary, 
            passes=15, 
            random_state=42, 
            alpha='auto'
        )
        
        coherence_model = CoherenceModel(
            model=lda_model, 
            texts=bow_list, 
            dictionary=dictionary, 
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        print(f"Coherence score for 78 topics: {coherence_score:.4f}")

        topic_sentiments = {}
        topic_counts = Counter()
        
        for i, doc_data in enumerate(processed_data):
            bow = corpus[i]
            topic_dist = lda_model.get_document_topics(bow)
            
            if topic_dist:
                dominant_topic = max(topic_dist, key=lambda x: x[1])[0]
                topic_counts[dominant_topic] += 1
                
                if dominant_topic not in topic_sentiments:
                    topic_sentiments[dominant_topic] = {
                        'compound_scores': [],
                        'classifications': [],
                        'positive_scores': [],
                        'negative_scores': [],
                        'keywords': []
                    }
                
                sentiment = doc_data['sentiment']
                topic_sentiments[dominant_topic]['compound_scores'].append(sentiment['compound'])
                topic_sentiments[dominant_topic]['classifications'].append(sentiment['classification'])
                topic_sentiments[dominant_topic]['positive_scores'].append(sentiment['positive'])
                topic_sentiments[dominant_topic]['negative_scores'].append(sentiment['negative'])
        
        for topic_id in range(78):
            try:
                words = [w for w, _ in lda_model.show_topic(topic_id, topn=8)]
                if topic_id in topic_sentiments:
                    topic_sentiments[topic_id]['keywords'] = words
            except Exception:
                if topic_id in topic_sentiments:
                    topic_sentiments[topic_id]['keywords'] = ["N/A"]

        # Calculate overall platform sentiment
        all_compound_scores = []
        all_classifications = []
        
        for doc_data in processed_data:
            all_compound_scores.append(doc_data['sentiment']['compound'])
            all_classifications.append(doc_data['sentiment']['classification'])
        
        overall_compound = sum(all_compound_scores) / len(all_compound_scores)
        classification_counts = Counter(all_classifications)

        print("\nOVERALL PLATFORM SENTIMENT ANALYSIS:")
        print(f"Average Compound Score: {overall_compound:.4f}")
        print(f"Positive documents: {classification_counts['positive']} ({classification_counts['positive']/len(processed_data)*100:.1f}%)")
        print(f"Negative documents: {classification_counts['negative']} ({classification_counts['negative']/len(processed_data)*100:.1f}%)")
        print(f"Neutral documents: {classification_counts['neutral']} ({classification_counts['neutral']/len(processed_data)*100:.1f}%)")

        if overall_compound > 0.05:
            overall_tone = "POSITIVE"
        elif overall_compound < -0.05:
            overall_tone = "NEGATIVE"
        else:
            overall_tone = "NEUTRAL"
        
        print(f"\nOverall Platform Tone: {overall_tone}")
        

        print("AVARAGE SENTIMENT ANALYSIS BY TOPIC (Top 15 Topics):")
        topic_avg_sentiments = []
        for topic_id, data in topic_sentiments.items():
            if data['compound_scores']:
                avg_compound = sum(data['compound_scores']) / len(data['compound_scores'])
                classification_dist = Counter(data['classifications'])
                total_docs = len(data['compound_scores'])
                
                topic_avg_sentiments.append({
                    'topic_id': topic_id,
                    'avg_compound': avg_compound,
                    'doc_count': total_docs,
                    'positive_pct': classification_dist['positive'] / total_docs * 100,
                    'negative_pct': classification_dist['negative'] / total_docs * 100,
                    'neutral_pct': classification_dist['neutral'] / total_docs * 100,
                    'keywords': ', '.join(data['keywords'][:5])
                })
        
        topic_avg_sentiments.sort(key=lambda x: x['doc_count'], reverse=True)
        top_topics = topic_avg_sentiments[:15]
        
        print(f"\n{'Topic':<6} {'Docs':<6} {'Avg Compound':<12} {'Positive %':<11} {'Negative %':<11} {'Neutral %':<11} {'Top Keywords'}")
        print("-" * 100)
        
        for topic_data in top_topics:
            topic_id = topic_data['topic_id']
            doc_count = topic_data['doc_count']
            avg_compound = topic_data['avg_compound']
            pos_pct = topic_data['positive_pct']
            neg_pct = topic_data['negative_pct']
            neutral_pct = topic_data['neutral_pct']
            keywords = topic_data['keywords']
            
            print(f"{topic_id:<6} {doc_count:<6} {avg_compound:<12.4f} {pos_pct:<11.1f} {neg_pct:<11.1f} {neutral_pct:<11.1f} {keywords}")
        
        if topic_avg_sentiments:
            most_positive = max(topic_avg_sentiments, key=lambda x: x['avg_compound'])
            most_negative = min(topic_avg_sentiments, key=lambda x: x['avg_compound'])
            
            print(f"\nMost Positive Topic: #{most_positive['topic_id']} (Score: {most_positive['avg_compound']:.4f})")
            print(f"  Keywords: {most_positive['keywords']}")
            print(f"  Documents: {most_positive['doc_count']}")
            
            print(f"\nMost Negative Topic: #{most_negative['topic_id']} (Score: {most_negative['avg_compound']:.4f})")
            print(f"  Keywords: {most_negative['keywords']}")
            print(f"  Documents: {most_negative['doc_count']}")

        return {
            'lda_model': lda_model,
            'processed_data': processed_data,
            'topic_sentiments': topic_sentiments,
            'overall_sentiment': {
                'compound': overall_compound,
                'classification_counts': classification_counts,
                'tone': overall_tone
            }
        }
    return

if __name__ == '__main__':
    main()