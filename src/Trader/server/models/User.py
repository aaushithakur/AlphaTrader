from bson import ObjectId
import json


class User:
  def __init__(self, _id, username, password, constants, portfolio, avatar=None, color=None, status='active',BUY_DATES={}):
    self._id = str(_id)
    self.username = username
    self.password = ""
    self.constants = constants
    self.portfolio = portfolio
    self.avatar = avatar
    self.color = color
    self.status = status
    self.BUY_DATES = BUY_DATES

  def is_authenticated(self):
    return True

  def is_active(self):
    return True

  def is_anonymous(self):
    return False

  def get_id(self):
    return self._id
  
  def to_dict(self):
    return {
        '_id': self._id,
        'username': self.username,
        'password': self.password,
        'constants': self.constants,
        'portfolio': self.portfolio,
        'avatar': self.avatar,
        'color': self.color,
        'status': self.status,
        'BUY_DATES': self.BUY_DATES,
    }

  def to_json(self):
    return json.dumps({
      "_id": self._id,
      "username": self.username,
      "password": self.password,
      "constants": self.constants,
      "portfolio": self.portfolio,
      "avatar": self.avatar,
      "status": self.status,
      "BUY_DATES": self.BUY_DATES
    })
    