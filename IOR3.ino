// Define Pins
const int Lpin = 2;
const int fixation = 15;
const int Rpin = 13;
const int leftButton = 14;
const int rightButton = 12;

unsigned long start = 0;
unsigned long reactionTime = 0;
bool taskStarted = false;
bool waitingForReaction = false;

// Cue and Target arrays 
int cueArray[2] = {2, 13};
int targetArray[2] = {2, 13};
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

      digitalWrite(Rpin, LOW);
      digitalWrite(Lpin, LOW);

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

    while (true); // Stop loop
  }

  previousLeftState = leftButtonState;
  previousRightState = rightButtonState;
}
