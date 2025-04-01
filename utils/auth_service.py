from typing import Optional
import bcrypt
import streamlit as st
import sqlite3
import os
from dataclasses import dataclass

@dataclass
class User:
    user_id: int
    username: str
    password_hash: str

class PasswordEncrypt:
    def encrypt(self, password):
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        return hashed_password.decode('utf-8')  # Store as string in DB

    def verify(self, current_password, db_password):
        return bcrypt.checkpw(current_password.encode('utf-8'), db_password.encode('utf-8'))

class PasswordRuleEngine:
    def is_valid(self, password):
        errors = []
        
        # Min length rule
        if len(password) < 8:
            errors.append("MinLengthRule")
            
        # Max length rule
        if len(password) > 26:
            errors.append("MaxLengthRule")
            
        # Uppercase rule
        if not any(char.isupper() for char in password):
            errors.append("UpperCaseRule")
            
        # Lowercase rule
        if not any(char.islower() for char in password):
            errors.append("LowerCaseRule")
            
        # Number rule
        if not any(char.isdigit() for char in password):
            errors.append("NumberRule")
            
        # Special character rule
        special_chars = "!@#$%^&*()"
        if not any(char in special_chars for char in password):
            errors.append("SpecialCharacterRule")
            
        if errors:
            return {"error": f"Password failed these rules: {', '.join(errors)}"}
        return True

class UserRepository:
    def __init__(self, db_path="data/users.db"):
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS Users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        conn.commit()
        conn.close()
        
    def get_by_username(self, username):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, username, password FROM Users WHERE username = ?", (username,))
        user_data = cursor.fetchone()
        conn.close()
        
        if user_data:
            return User(user_id=user_data[0], username=user_data[1], password_hash=user_data[2])
        return None
        
    def get_by_id(self, user_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, username, password FROM Users WHERE user_id = ?", (user_id,))
        user_data = cursor.fetchone()
        conn.close()
        
        if user_data:
            return User(user_id=user_data[0], username=user_data[1], password_hash=user_data[2])
        return None
        
    def add(self, username, password_hash):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO Users (username, password) VALUES (?, ?)",
                (username, password_hash)
            )
            user_id = cursor.lastrowid
            conn.commit()
            return User(user_id=user_id, username=username, password_hash=password_hash)
        except sqlite3.IntegrityError:
            # Username already exists
            return {"error": "Username already exists"}
        finally:
            conn.close()

class UserService:
    def __init__(self):
        self._repository = UserRepository()
        self.password_rule_engine = PasswordRuleEngine()
        self.password_encrypt = PasswordEncrypt()
        
    def create_user(self, username, password):
        # Validate password
        password_validation = self.password_rule_engine.is_valid(password)
        if isinstance(password_validation, dict):
            return password_validation
            
        # Encrypt password
        hashed_password = self.password_encrypt.encrypt(password)
        
        # Add user to repository
        return self._repository.add(username, hashed_password)
        
    def login_user(self, username, password):
        user = self._repository.get_by_username(username)
        
        if not user:
            return {"error": "User not found"}
            
        if not self.password_encrypt.verify(password, user.password_hash):
            return {"error": "Invalid password"}
            
        return {"success": "Login successful", "user": user}