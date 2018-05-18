#include <SoftwareSerial.h>

//----- Serial Config -----//
const byte buffSize = 64;
char inputBuffer[buffSize];
const char startMarker = '<';
const char endMarker = '>';
byte bytesRecvd = 0;
boolean readInProgress = false;
boolean newDataFromPC = false;
char messageFromPC[buffSize] = {0};
//----- End of Serial Config -----//

int speed;
char ignore;

void forward(int pwm) {
  analogWrite(9, pwm);
  digitalWrite(5, HIGH);
  digitalWrite(4, LOW);

  analogWrite(10, pwm);
  digitalWrite(8, HIGH);
  digitalWrite(7, LOW);

}

void brake() {
  digitalWrite(9, HIGH);
  digitalWrite(5, HIGH);
  digitalWrite(4, HIGH);

  digitalWrite(10, HIGH);
  digitalWrite(8, HIGH);
  digitalWrite(7, HIGH);

}

void backward(int pwm) {
  analogWrite(9, pwm);
  digitalWrite(5, LOW);
  digitalWrite(4, HIGH);

  analogWrite(10, pwm);
  digitalWrite(8, LOW);
  digitalWrite(7, HIGH);
}

void getDataFromPC() {
  // receive data from PC and save it into inputBuffer
  if (Serial.available() > 0) {
    char x = Serial.read();
    // the order of these IF clauses is significant
    if (x == endMarker) {
      readInProgress = false;
      newDataFromPC = true;
      inputBuffer[bytesRecvd] = 0;
      parseData();
    }
    if (readInProgress) {
      inputBuffer[bytesRecvd] = x;
      bytesRecvd ++;
      if (bytesRecvd == buffSize) {
        bytesRecvd = buffSize - 1;
      }
    }
    if (x == startMarker) {
      bytesRecvd = 0;
      readInProgress = true;
    }
  }
}

void parseData() {
  char * strtokIndx; // this is used by strtok() as an index
  strtokIndx = strtok(inputBuffer, ",");
  speed = atof(strtokIndx);
  strtokIndx = strtok(NULL, ",");
  ignore = atof(strtokIndx);
}

void replyToPC() {
  if (newDataFromPC) {
    newDataFromPC = false;
    Serial.print("<speed: ");
    Serial.print(speed * 10);
    Serial.print(" ignore: ");
    Serial.print(ignore);
    Serial.println(">");
  }
}

void setup() {
  pinMode(A3, OUTPUT); // STANDBY
  digitalWrite(A3, LOW);

  pinMode(10, OUTPUT); // PWM A
  pinMode(8, OUTPUT); // MOTOR A
  pinMode(7, OUTPUT);

  pinMode(9, OUTPUT); // PWM B
  pinMode(5, OUTPUT);  // MOTOR B
  pinMode(4, OUTPUT);

  pinMode(LED_BUILTIN, OUTPUT);

  Serial.begin(9600);
  delay(10);
  speed = 0;
  Serial.println("<Arduino is ready>");

  digitalWrite(A3, HIGH);
}

// the loop function runs over and over again forever
void loop() {
  getDataFromPC();
  replyToPC();

  if (speed < 12 && speed > 0) {
        forward(speed*10);
  };

  if (speed > -12 && speed < 0) {
        backward(speed*10);
  };
  
  if (speed == 0 || speed > 12 || speed < -12) {
    brake();
  };

}

