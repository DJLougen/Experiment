#include <WiFi.h>
#include <HTTPClient.h>

// WiFi credentials
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* serverName = "http://192.168.1.100:5000/upload";  // Flask server on your Mac

// Define Pins
// Define Pins
const int Lpin = 19;
const int fixation = 23;
const int Rpin = 18;
const int leftButton = 14;
const int rightButton = 12;

unsigned long start = 0;
unsigned long reactionTime = 0;
bool taskStarted = false;
bool waitingForReaction = false;

// Cue and Target arrays 
int cueArray[2] = {Lpin, Rpin};
int targetArray[2] = {Lpin, Rpin};
int ISI[3] = {100, 300, 750};

// Trial data
const int maxTrials = 10;
unsigned long reactionTimes[maxTrials];
int cueSides[maxTrials];
int targetSides[maxTrials];
int gaps[maxTrials];
String validity[maxTrials];
String responseSides[maxTrials];
int trialCount = 0;

// Helper function to convert pin to "L"/"R"
String pinToSide(int pin) {
  if (pin == Lpin) return "L";
  if (pin == Rpin) return "R";
  return "Unknown";
}

void setup() {
  Serial.begin(115200);
  pinMode(Lpin, OUTPUT);
  pinMode(fixation, OUTPUT);
  pinMode(Rpin, OUTPUT);
  pinMode(leftButton, INPUT_PULLUP);
  pinMode(rightButton, INPUT_PULLUP);

  // Connect to WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected! IP Address: ");
  Serial.println(WiFi.localIP());

  // Ensure all outputs are LOW at idle
  digitalWrite(Lpin, LOW);
  digitalWrite(Rpin, LOW);
  digitalWrite(fixation, LOW);
}

void loop() {
  static int previousLeftState = HIGH;
  static int previousRightState = HIGH;
  int leftButtonState = digitalRead(leftButton);
  int rightButtonState = digitalRead(rightButton);

  if (trialCount < maxTrials) {
    if (!taskStarted && ((leftButtonState == LOW && previousLeftState == HIGH) || (rightButtonState == LOW && previousRightState == HIGH))) {
      taskStarted = true;
      previousLeftState = leftButtonState;
      previousRightState = rightButtonState;

      int cueSide = cueArray[random(0, 2)];
      int targetSide = targetArray[random(0, 2)];
      int gap = ISI[random(0, 3)];

      // Save metadata
      cueSides[trialCount] = cueSide;
      targetSides[trialCount] = targetSide;
      gaps[trialCount] = gap;
      validity[trialCount] = (cueSide == targetSide) ? "valid" : "invalid";

      digitalWrite(fixation, HIGH); 
      delay(750);
      digitalWrite(fixation, LOW);

      digitalWrite(cueSide, HIGH);
      delay(100);
      digitalWrite(cueSide, LOW);

      digitalWrite(fixation, HIGH);
      delay(50);
      digitalWrite(fixation, LOW);
      delay(gap);

      digitalWrite(targetSide, HIGH);
      start = millis();
      waitingForReaction = true;
    }

    if (waitingForReaction && ((leftButtonState == LOW && previousLeftState == HIGH) || (rightButtonState == LOW && previousRightState == HIGH))) {
      unsigned long end = millis();
      reactionTime = end - start;
      reactionTimes[trialCount] = reactionTime;
      if (leftButtonState == LOW && previousLeftState == HIGH) {
        responseSides[trialCount] = "L";
      } else if (rightButtonState == LOW && previousRightState == HIGH) {
        responseSides[trialCount] = "R";
      } else {
        responseSides[trialCount] = "Unknown";
      }
      trialCount++;

      // Only turn off the target LED that was lit
      digitalWrite(targetSides[trialCount-1], LOW);

      taskStarted = false;
      waitingForReaction = false;
    }

  } else {
    // Output CSV once after 10 trials
    Serial.println("\nTrial,CueSide,TargetSide,Gap(ms),CueValidity,ReactionTime(ms),ResponseSide");
    for (int i = 0; i < maxTrials; i++) {
      Serial.print(i + 1);
      Serial.print(",");
      Serial.print(pinToSide(cueSides[i]));
      Serial.print(",");
      Serial.print(pinToSide(targetSides[i]));
      Serial.print(",");
      Serial.print(gaps[i]);
      Serial.print(",");
      Serial.print(validity[i]);
      Serial.print(",");
      Serial.print(reactionTimes[i]);
      Serial.print(",");
      Serial.println(responseSides[i]);
    }

    // Prepare CSV string
    String csv = "Trial,CueSide,TargetSide,Gap(ms),CueValidity,ReactionTime(ms),ResponseSide\n";
    for (int i = 0; i < maxTrials; i++) {
      csv += String(i + 1) + "," + pinToSide(cueSides[i]) + "," + pinToSide(targetSides[i]) + "," + String(gaps[i]) + "," + validity[i] + "," + String(reactionTimes[i]) + "," + responseSides[i] + "\n";
    }

    // Send CSV to Flask server
    if (WiFi.status() == WL_CONNECTED) {
      HTTPClient http;
      http.begin(serverName);
      String boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW";
      String contentType = "multipart/form-data; boundary=" + boundary;
      http.addHeader("Content-Type", contentType);
      String body = "--" + boundary + "\r\n";
      body += "Content-Disposition: form-data; name=\"file\"; filename=\"reaction_data.csv\"\r\n";
      body += "Content-Type: text/csv\r\n\r\n";
      body += csv;
      body += "\r\n--" + boundary + "--\r\n";
      int httpResponseCode = http.POST(body);
      Serial.print("Upload response code: ");
      Serial.println(httpResponseCode);
      http.end();
    } else {
      Serial.println("WiFi not connected, could not upload CSV");
    }

    while (true); // Stop loop
  }

  previousLeftState = leftButtonState;
  previousRightState = rightButtonState;
}
