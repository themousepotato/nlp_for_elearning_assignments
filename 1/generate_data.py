#!/usr/bin/python
#-*- coding: utf-8 -*-

from tqdm import tqdm
import tweepy
import time
import csv
import os
import re


CONSUMER_KEY = '6nn9gH6GheUv9pWaHILxKikRF'
CONSUMER_SECRET = 'DqaLEdq9BSA6djRYJJv9wcv5tEGHUQHVpI9O1sTU1Ky1vZSgLV'
ACCESS_TOKEN = '1005448420780199936-sOUCqjKZi2NdPJe9OdE5hKrFJmPbBA'
ACCESS_TOKEN_SECRET = 'C17bY7ofjvTYQYMK6mvlukbq98stJgQiBE9y1AGjY4jxv'
TWEET_IDS_FILE = 'democratic-candidate-timelines.txt'
CORPUS_FILE = 'corpus.txt'


def create_training_set():
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    with open(TWEET_IDS_FILE, 'r') as f:
        ids = f.read().split()

    training_dataset = []
    with tqdm(total=len(ids)) as t:
        for tweet_id in ids:
            try:
                fetched_tweet = api.get_status(tweet_id)
                #print('Tweet fetched with id={0}'.format(tweet_id))
                tweet_text = fetched_tweet.text

                # Remove urls and hashtags
                urls = re.findall(r'(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+', tweet_text)
                hashtags = re.findall(r'\B#\w*[a-zA-Z]+\w*', tweet_text)
                for garbage in urls + hashtags:
                    tweet_text = tweet_text.replace(garbage, '')

                training_dataset.append(tweet_text)
                t.update()
                #time.sleep(0.001)

            except Exception as e:
                print(e)
                print('Failed to fetch tweet with id={0}'.format(tweet_id))
                continue

    with open(CORPUS_FILE, 'a') as f:
        for tweet in training_dataset:
            try:
                f.write(tweet+'\n')
            except Exception as e:
                print(e)


if __name__ == '__main__':
    create_training_set()
