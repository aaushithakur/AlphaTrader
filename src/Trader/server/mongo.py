import os
from pymongo import MongoClient, ReturnDocument
from bson import ObjectId
from server import models
# from env.config import get_env

def get_database():
 
   # Provide the mongodb atlas url to connect python to mongodb using pymongo
   #  CONNECTION_STRING = get_env("DATABASE_URL")
   CONNECTION_STRING = 'mongodb://root:akshat@mongo:27017'
   # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
   client = MongoClient(CONNECTION_STRING)
 
   # Create the database for our example (we will use the same database throughout the tutorial
   return client['crypto']
  
# This is added so that many files can reuse the function get_database()
if __name__ == "__main__":
   # Get the database
   dbname = get_database()

def find_user(username, password):
  dbname = get_database()
  users = dbname["users"]
  user = users.find_one({ "username": username, "password": password })
  # If the user doesn't have constants, load those constants
  # if ('constants' not in user):
  #   user = users.find_one_and_update({"username": username, "password": password }, 
  #     { 
  #       '$set': { 
  #         'constants': default_const,
  #         'portfolio': default_portfolio
  #       }
  #     }, return_document=ReturnDocument.AFTER)
  return models.User(**user)

def get_users():
  dbname = get_database()
  users = dbname["users"]
  users = list(users.find())
  return users

def get_user(user_id):
  dbname = get_database()
  users = dbname["users"]
  user = users.find_one({ "_id": ObjectId(user_id )})
  return models.User(**user)

def update_user(current_user, input_user):
  dbname = get_database()
  users = dbname["users"]
  user = users.find_one_and_update({ "_id": ObjectId(current_user._id)}, { "$set": { **input_user } },  return_document=ReturnDocument.AFTER)
  return models.User(**user)

def get_portfolio(user):
  db_user = get_user(user._id)
  return db_user.portfolio

def set_portfolio(user, portfolio):
  return update_user(user, { "portfolio": portfolio })

def update_buy_dates(username, buy_dates):
    dbname = get_database()
    users = dbname["users"]
    users.update_one({"username": username}, {"$set": {"BUY_DATES": buy_dates}})
  
def update_bid_price_dict(username, bid_price_dict):
    dbname = get_database()
    users = dbname["users"]
    users.update_one({"username": username}, {"$set": {"BID_PRICE_DICT": bid_price_dict}})
