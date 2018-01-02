#include <SoftwareSerial.h>
#include <SabertoothSimplified.h>
#include <NewPing.h>

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

int leftSpeed;
int rightSpeed;
bool do_ping;
byte distance;

#define TRIGGER_PIN  A1  // Arduino pin tied to trigger pin on the ultrasonic sensor.
#define ECHO_PIN     A2  // Arduino pin tied to echo pin on the ultrasonic sensor.
#define MAX_DISTANCE 21

NewPing sonar(TRIGGER_PIN, ECHO_PIN, MAX_DISTANCE);

SoftwareSerial SWSerial(NOT_A_PIN, 3); // RX on no pin (unused), TX on pin 11 (to S1).
SabertoothSimplified ST(SWSerial); // Use SWSerial as the serial port.

void setup()
{
  Serial.begin(9600);
  delay(10);
  SWSerial.begin(9600);
  delay(10);
  leftSpeed = 0;
  rightSpeed = 0;
  Serial.println("<Arduino is ready>");
  do_ping = true;
}

void loop()
{
  getDataFromPC();
  replyToPC();
  //distance = sonar.ping_cm();
  //Serial.println(distance);
  if (leftSpeed < 6 && rightSpeed < 6) {
    //do_ping = !do_ping;
    if (do_ping == true) {
      distance = sonar.ping_cm();
      //Serial.println(distance);
      if (distance!=0 && distance < 19) {
        //Serial.println("ESTOP");
        ST.motor(1, 0);  //127 is max power
        ST.motor(2, 0);
      } else {
        //Serial.println("moving");
        ST.motor(1, leftSpeed * 10);  //127 is max power
        ST.motor(2, rightSpeed * 10);
      }
    }
  }
}

void getDataFromPC() {
    // receive data from PC and save it into inputBuffer
  if(Serial.available() > 0) {
    char x = Serial.read();
      // the order of these IF clauses is significant
    if (x == endMarker) {
      readInProgress = false;
      newDataFromPC = true;
      inputBuffer[bytesRecvd] = 0;
      parseData();
    }
    if(readInProgress) {
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
  strtokIndx = strtok(inputBuffer,",");
  leftSpeed = atof(strtokIndx);
  strtokIndx = strtok(NULL, ","); 
  rightSpeed = atof(strtokIndx);
}

void replyToPC() {
  if (newDataFromPC) {
    newDataFromPC = false;
    Serial.print("<Left: ");
    Serial.print(leftSpeed * 11);
    Serial.print(" right: ");
    Serial.print(rightSpeed * 10);
    Serial.println(">");
  }
}

