import sys
import sqlite3
import subprocess
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox


class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Login")
        self.setFixedSize(400, 300)

        # Create GUI components
        self.username_label = QLabel("Username:")
        self.username_input = QLineEdit()
        self.password_label = QLabel("Password:")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)

        self.login_button = QPushButton("Login")
        self.signup_button = QPushButton("Sign Up")

        # Layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.username_label)
        self.layout.addWidget(self.username_input)
        self.layout.addWidget(self.password_label)
        self.layout.addWidget(self.password_input)
        self.layout.addWidget(self.login_button)
        self.layout.addWidget(self.signup_button)
        self.setLayout(self.layout)

        # Database connection
        self.conn = sqlite3.connect("users.db")
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)"
        )

        # Connect buttons
        self.login_button.clicked.connect(self.login)
        self.signup_button.clicked.connect(self.signup)

    def login(self):
        """Verify user credentials and launch the Flask app."""
        username = self.username_input.text()
        password = self.password_input.text()

        # Query the database for the user
        self.cursor.execute(
            "SELECT * FROM users WHERE username=? AND password=?", (username, password)
        )
        user = self.cursor.fetchone()

        if user:
            # Launch Flask app if login is successful
            subprocess.Popen(["python", "app.py"])
            self.close()
        else:
            QMessageBox.warning(self, "Login Failed", "Invalid username or password.")

    def signup(self):
        """Add a new user to the database."""
        username = self.username_input.text()
        password = self.password_input.text()

        # Check if the username already exists
        self.cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        user = self.cursor.fetchone()

        if user:
            QMessageBox.warning(self, "Sign Up Failed", "Username already exists.")
        else:
            self.cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            self.conn.commit()
            QMessageBox.information(self, "Sign Up Successful", "User registered successfully!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    login_window = LoginWindow()
    login_window.show()
    sys.exit(app.exec_())
