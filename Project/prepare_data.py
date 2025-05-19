import os
import glob
import wget
import tarfile
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
import re
from bs4 import BeautifulSoup
import pickle
from collections import defaultdict
from tqdm import tqdm
class IMDbDataPreparation:
    def __init__(self, base_dir='./data'):
        """Initialize IMDb data preparation class"""
        self.base_dir = base_dir
        self.imdb_dir = os.path.join(base_dir, 'aclImdb')
        self.data_dir = os.path.join(base_dir, 'processed')
        self.cache_dir = os.path.join("cache", "sentiment_analysis")
        
        # Create necessary directories
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def download_data(self):
        """Download IMDb dataset if not exists"""
        if not os.path.exists(self.imdb_dir):
            print("Downloading IMDb dataset...")
            # Download the dataset
            url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
            filename = os.path.join(self.base_dir, "aclImdb_v1.tar.gz")
            wget.download(url, filename)
            
            # Extract the dataset
            print("\nExtracting dataset...")
            with tarfile.open(filename, 'r:gz') as tar:
                tar.extractall(path=self.base_dir)
                
            # Remove the compressed file
            os.remove(filename)
        else:
            print("IMDb dataset already exists.")

    def read_imdb_data(self):
        """Read IMDb data from files"""
        data = {}
        labels = {}
        
        for data_type in ['train', 'test']:
            data[data_type] = {}
            labels[data_type] = {}
            
            for sentiment in ['pos', 'neg']:
                data[data_type][sentiment] = []
                labels[data_type][sentiment] = []
                
                path = os.path.join(self.imdb_dir, data_type, sentiment, '*.txt')
                files = glob.glob(path)
                
                for f in tqdm(files, desc="nhanh len"):
                    with open(f, encoding='utf-8') as review:
                        data[data_type][sentiment].append(review.read())
                        labels[data_type][sentiment].append(1 if sentiment == 'pos' else 0)
                        
                assert len(data[data_type][sentiment]) == len(labels[data_type][sentiment]), \
                        f"{data_type}/{sentiment} data size does not match labels size"
                    
        return data, labels

    def prepare_imdb_data(self, data, labels):
        """Prepare training and test sets"""
        data_train = data['train']['pos'] + data['train']['neg']
        data_test = data['test']['pos'] + data['test']['neg']
        labels_train = labels['train']['pos'] + labels['train']['neg']
        labels_test = labels['test']['pos'] + labels['test']['neg']
        
        # Shuffle data
        data_train, labels_train = shuffle(data_train, labels_train)
        data_test, labels_test = shuffle(data_test, labels_test)

        
        return data_train, data_test, labels_train, labels_test

    def review_to_words(self, review):
        """Convert review to words"""
        nltk.download("stopwords", quiet=True)
        
        text = BeautifulSoup(review, "html.parser").get_text()
        text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
        words = text.split()
        words = [w for w in words if w not in stopwords.words("english")]
        words = [PorterStemmer().stem(w) for w in words]
        
        return words

    def build_dict(self, data, vocab_size=5000):
        """Build vocabulary dictionary"""
        word_count = defaultdict(int)
        
        for sentence in data:
            for word in sentence:
                word_count[word] += 1
                
        sorted_words = sorted(word_count, key=word_count.get, reverse=True)
        
        word_dict = {}
        for idx, word in enumerate(sorted_words[:vocab_size-2]):
            word_dict[word] = idx + 2
            
        return word_dict

    def convert_and_pad(self, word_dict, sentence, pad=500):
            """Convert words to integers and pad sequences"""
            NOWORD = 0  # Đại diện cho 'no word'
            INFREQ = 1  # Đại diện cho từ không có trong word_dict
            
            working_sentence = [NOWORD] * pad
            
            for word_index, word in enumerate(sentence[:pad]):
                if word in word_dict:
                    working_sentence[word_index] = word_dict[word]
                else:
                    working_sentence[word_index] = INFREQ
                    
            return working_sentence, min(len(sentence), pad)

    def convert_and_pad_data(self, word_dict, data, pad=500):
        """Convert and pad all reviews in dataset"""
        result = []
        lengths = []
        
        for sentence in data:
            converted, leng = self.convert_and_pad(word_dict, sentence, pad)
            result.append(converted)
            lengths.append(leng)
            
        return np.array(result), np.array(lengths)

    def transform_and_save_data(self, train_X, test_X, train_y, test_y, word_dict):
        """Transform data and save to files"""
        print("Transforming and saving data...")
        
        # Convert and pad sequences
        train_X, train_X_len = self.convert_and_pad_data(word_dict, train_X)
        test_X, test_X_len = self.convert_and_pad_data(word_dict, test_X)
        
        # Save training data
        pd.concat([
            pd.DataFrame(train_y), 
            pd.DataFrame(train_X_len), 
            pd.DataFrame(train_X)
        ], axis=1).to_csv(
            os.path.join(self.data_dir, 'train.csv'), 
            header=False, 
            index=False
        )
        
        # Save test data  
        pd.concat([
            pd.DataFrame(test_y),
            pd.DataFrame(test_X_len),
            pd.DataFrame(test_X)
        ], axis=1).to_csv(
            os.path.join(self.data_dir, 'test.csv'), 
            header=False, 
            index=False
        )
        
        print("Data transformation completed!")
        return train_X, test_X, train_y, test_y, train_X_len, test_X_len

    def prepare_data(self, cache_file="preprocessed_data.pkl"):
        """Main function to prepare all data"""
        print("Starting data preparation...")
        
        # Step 1: Download data if needed
        self.download_data()
        
        # Step 2: Read and prepare data
        data, labels = self.read_imdb_data()
        train_X, test_X, train_y, test_y = self.prepare_imdb_data(data, labels)

        train_X = train_X[:250]
        train_y = train_y[:250]
        test_X = test_X[:250]
        test_y = test_y[:250]
                
        # Step 3: Process reviews to words
        # Try to load from cache first
        cache_path = os.path.join(self.cache_dir, cache_file)
        if os.path.exists(cache_path):
            print("Loading preprocessed data from cache...")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                train_X = cache_data['words_train']
                test_X = cache_data['words_test']
                train_y = cache_data['labels_train']
                test_y = cache_data['labels_test']
        else:
            print("Processing reviews to words...")
            train_X = [self.review_to_words(review) for review in train_X]
            test_X = [self.review_to_words(review) for review in test_X]
            
            # Save to cache
            cache_data = {
                'words_train': train_X,
                'words_test': test_X,
                'labels_train': train_y,
                'labels_test': test_y
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        
        # Step 4: Build vocabulary
        word_dict = self.build_dict(train_X)
        
        # Save word dictionary
        with open(os.path.join(self.data_dir, 'word_dict.pkl'), 'wb') as f:
            pickle.dump(word_dict, f)

        # Step 5: Transform and save data
        train_X, test_X, train_y, test_y, train_X_len, test_X_len = self.transform_and_save_data(
            train_X, test_X, train_y, test_y, word_dict
        )
        
        print("Data preparation completed!")
        return {
            'train_X': train_X, 
            'test_X': test_X,
            'train_y': train_y,
            'test_y': test_y,
            'train_X_len': train_X_len,
            'test_X_len': test_X_len,
            'word_dict': word_dict
        }, self.data_dir