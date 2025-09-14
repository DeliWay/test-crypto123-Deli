# models.py (новый файл)
from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Подписка
    subscription_type = db.Column(db.String(20), default='free')  # free, premium
    subscription_end = db.Column(db.DateTime)
    payment_id = db.Column(db.String(100))  # ID из платежной системы

    def has_premium(self):
        if self.subscription_type == 'premium' and self.subscription_end > datetime.utcnow():
            return True
        return False