import bcrypt
import sqlite3

# Initialize the database (Creates users table if it doesn't exist)
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)''')
    conn.commit()
    conn.close()

# Hash the password before storing it
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# Verify a password against the stored hash
def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode(), hashed_password.encode())

# Register a new user (returns False if username already exists)
def register_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    hashed_password = hash_password(password)
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        conn.close()
        return True  # Registration successful
    except sqlite3.IntegrityError:
        return False  # Username already exists

# Authenticate user login (returns True if login is successful)
def authenticate_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        return verify_password(password, result[0])
    return False  # User not found or wrong password

# Initialize the database when the app starts
init_db()
