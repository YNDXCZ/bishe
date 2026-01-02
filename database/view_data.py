import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.db_manager import DatabaseManager
import config

def view_data():
    print(f"Connecting to MySQL Database '{config.DB_NAME}' at {config.DB_HOST}...")
    db = DatabaseManager()
    
    if not db.conn:
        print("Failed to connect to database.")
        return

    cursor = db.conn.cursor(dictionary=True)

    # 1. View Users
    print("\n=== Table: users ===")
    try:
        cursor.execute("SELECT * FROM users")
        users = cursor.fetchall()
        if not users:
            print("(No users found)")
        else:
            print(f"{'ID':<5} {'Username':<20} {'Created At'}")
            print("-" * 45)
            for user in users:
                print(f"{user['id']:<5} {user['username']:<20} {user['created_at']}")
    except Exception as e:
        print(f"Error querying users: {e}")

    # 2. View Logs (Last 10)
    print("\n=== Table: posture_logs (Last 10 Reocrds) ===")
    try:
        cursor.execute("""
            SELECT l.id, u.username, l.posture_type, l.duration_seconds, l.timestamp 
            FROM posture_logs l
            JOIN users u ON l.user_id = u.id
            ORDER BY l.timestamp DESC 
            LIMIT 10
        """)
        logs = cursor.fetchall()
        if not logs:
            print("(No logs found)")
        else:
            print(f"{'ID':<5} {'User':<15} {'Type':<10} {'Duration(s)':<12} {'Timestamp'}")
            print("-" * 65)
            for log in logs:
                print(f"{log['id']:<5} {log['username']:<15} {log['posture_type']:<10} {log['duration_seconds']:<12} {log['timestamp']}")
    except Exception as e:
        print(f"Error querying logs: {e}")

if __name__ == "__main__":
    view_data()
