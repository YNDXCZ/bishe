import mysql.connector
from mysql.connector import Error
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def setup_database():
    print(f"Connecting to MySQL at {config.DB_HOST} with user '{config.DB_USER}'...")
    
    # 1. Create Database
    try:
        # Connect to MySQL Server (no DB selected yet)
        conn = mysql.connector.connect(
            host=config.DB_HOST,
            user=config.DB_USER,
            password=config.DB_PASSWORD
        )
        if conn.is_connected():
            cursor = conn.cursor()
            print(f"Creating Database '{config.DB_NAME}'...")
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {config.DB_NAME}")
            print("Database created (or already exists).")
            conn.close()
    except Error as e:
        print(f"CRITICAL ERROR: Could not connect to MySQL or create database.\nError: {e}")
        return

    # 2. Create Tables
    try:
        # Connect to the specific database
        conn = mysql.connector.connect(
            host=config.DB_HOST,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            database=config.DB_NAME
        )
        cursor = conn.cursor()
        
        print("Creating Table 'users'...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        print("Creating Table 'posture_logs'...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS posture_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            posture_type VARCHAR(20),
            duration_seconds FLOAT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """)
        
        # 3. Insert Default User
        print("Ensuring default user 'admin' exists...")
        cursor.execute("INSERT IGNORE INTO users (username) VALUES ('admin')")
        conn.commit()
        
        print("\nSUCCESS! Database setup complete.")
        print(f"You should now see database '{config.DB_NAME}' in Navicat.")
        
    except Error as e:
        print(f"Error setting up tables: {e}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            conn.close()

if __name__ == "__main__":
    setup_database()
