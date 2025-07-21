from flask import Flask, render_template, send_from_directory, request, send_from_directory
import socket
import os

app = Flask(__name__)
ESP32_IP = "192.168.1.113"
ESP32_PORT = 5000  # or whatever port you're using
CSV_FILE = "reaction_data.csv"

# Check if ESP32 is online
def is_esp32_connected():
    try:
        with socket.create_connection((ESP32_IP, ESP32_PORT), timeout=5):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False

@app.route('/')
def index():
    connected = is_esp32_connected()
    file_ready = os.path.exists(CSV_FILE)
    return render_template('index.html', connected=connected, file_ready=file_ready)

@app.route('/download')
def download():
    return send_from_directory(directory='.', path=CSV_FILE, as_attachment=True)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if file:
        file.save(CSV_FILE)
        return 'Upload successful', 200
    return 'No file uploaded', 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
