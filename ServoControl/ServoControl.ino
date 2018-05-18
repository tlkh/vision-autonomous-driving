#include <SoftwareSerial.h>

/* Sweep
 by BARRAGAN <http://barraganstudio.com>
 This example code is in the public domain.

 modified 8 Nov 2013
 by Scott Fitzgerald
 http://www.arduino.cc/en/Tutorial/Sweep
*/

#include <Servo.h>

Servo servo_front_right;
Servo servo_front_left;

Servo servo_back_right;
Servo servo_back_left;

int offset_front_right = 10;
int offset_front_left = 0;
int offset_back_right = 10;
int offset_back_left = 15;



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

int angle;
char ignore;

void posi(int angle) {
  servo_front_right.write(angle + offset_front_right);
  servo_front_left.write(angle + offset_front_left);
  
  servo_back_right.write(angle + offset_back_right);
  servo_back_left.write(angle + offset_back_left);

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
  angle = atof(strtokIndx);
  strtokIndx = strtok(NULL, ",");
  ignore = atof(strtokIndx);
}

void replyToPC() {
  if (newDataFromPC) {
    newDataFromPC = false;
    Serial.print("<angle: ");
    Serial.print(angle);
    Serial.print(" ignore: ");
    Serial.print(ignore);
    Serial.println(">");
  }
}

void setup() {
  
  servo_front_right.attach(10);
  servo_front_left.attach(11);
  
  servo_back_right.attach(9);
  servo_back_left.attach(6);

  Serial.begin(9600);
  delay(10);
  Serial.println("<Arduino is ready>");
  
  swing_swing();
  center();
  
}

void loop() {

  getDataFromPC();
  replyToPC();

  if (angle < 181 && angle > 0) {
        posi(angle);
  };
  
 // if (angle == 0 || speed > 12 || speed < -12) {
 //   brake();
 // };
}

void center() {
  servo_front_right.write(90 + offset_front_right);
  servo_front_left.write(90 + offset_front_left);
  
  servo_back_right.write(90 + offset_back_right);
  servo_back_left.write(90 + offset_back_left);
}

void swing_swing() {
  servo_front_right.write(45);
  servo_front_left.write(45);
  servo_back_right.write(45);
  servo_back_left.write(45);
  delay(500);
  servo_front_right.write(135);
  servo_front_left.write(135);
  servo_back_right.write(135);
  servo_back_left.write(135);
  delay(500);

  
}
