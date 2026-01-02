import mysql.connector
from mysql.connector import Error

class DatabaseManager:
    def __init__(self, host='localhost', database='posture_health', user='root', password=''):
        self.config = {
            'host': host,
            'database': database,
            'user': user,
            'password': password
        }
        self.conn = None
        self.init_db()

    def connect(self):
        try:
            self.conn = mysql.connector.connect(**self.config)
            if self.conn.is_connected():
                return self.conn
        except Error as e:
            # Handle case where database doesn't exist yet
             pass
        return None

    def create_database(self):
        try:
            temp_config = self.config.copy()
            del temp_config['database']
            conn = mysql.connector.connect(**temp_config)
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.config['database']}")
            conn.close()
        except Error as e:
            print(f"Error creating database: {e}")

    def init_db(self):
        self.create_database()
        try:
            self.conn = mysql.connector.connect(**self.config)
            cursor = self.conn.cursor()
            
            # Read schema from file if running from project root
            # Or define here. Let's define simple creates here for robustness
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS posture_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                posture_type VARCHAR(10),
                duration_seconds FLOAT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """)
            
            self.conn.commit()
        except Error as e:
            print(f"Error initializing tables: {e}")

    def add_user(self, username):
        try:
            cursor = self.conn.cursor()
            cursor.execute("INSERT IGNORE INTO users (username) VALUES (%s)", (username,))
            self.conn.commit()
            
            cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
            return cursor.fetchone()[0]
        except Error as e:
            print(f"Error adding user: {e}")
            return None

    def log_posture(self, user_id, posture_type, duration):
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO posture_logs (user_id, posture_type, duration_seconds) VALUES (%s, %s, %s)",
                (user_id, posture_type, duration)
            )
            self.conn.commit()
        except Error as e:
            print(f"Error logging posture: {e}")

    def get_stats(self, user_id, days=7):
        # Return stats for charts
        try:
            cursor = self.conn.cursor(dictionary=True)
            # Simple daily aggregation
            query = """
            SELECT DATE(timestamp) as date, posture_type, SUM(duration_seconds) as total_duration 
            FROM posture_logs 
            WHERE user_id = %s AND timestamp >= NOW() - INTERVAL %s DAY
            GROUP BY date, posture_type
            """
            cursor.execute(query, (user_id, days))
            return cursor.fetchall()
        except Error as e:
            print(f"Error getting stats: {e}")
            return []

if __name__ == "__main__":
    # Test
    db = DatabaseManager(password='123456') # Assumption on password? User needs to configure.
    # We will assume blank or standard, and let user detailed layout configure it.
    # Actually, for safety, I will try catch or let user change it.
    pass
