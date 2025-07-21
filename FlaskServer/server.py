## .venv\Scripts\activate

#Import necessary libraries
from flask import Flask, render_template
import socket

ESP32_IP = "192.168.1.100"  # Replace with ESP32 IP address
ESP32_PORT = 80

#Create a Flask application instance
app = Flask(__name__)

#Check if ESP32 is connected
def is_esp32_connected():
    try:
        with socket.create_connection((ESP32_IP, ESP32_PORT), timeout=5): # Establish a connection to the ESP32
            return True # Connection successful
    except (socket.timeout, ConnectionRefusedError):
        return False # Connection failed
  
#Route for the main page
@app.route('/')
def index():
    connected = is_esp32_connected()
    print("Server is running, ESP32 connected: {}".format(connected)) #Print Status
    return render_template('index.html', connected=connected) #Render index.html with connection status


if __name__ == '__main__':
    app.run(debug=True) # Run the Flask application
    # Set debug=True for development, change to False in production

