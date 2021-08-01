#include <SPI.h>

int CSB1 = 10;//set CSB port

unsigned int x_lsb = 0x02;
unsigned int y_lsb = 0x04;
unsigned int z_lsb = 0x06;//registers for reading acc

unsigned long currentMicros = 0;    // stores the value of micros() in each iteration of loop()

void setup() {
pinMode(CSB1, OUTPUT);
Serial.begin(921600);
SPI.begin();
SPI.setClockDivider(SPI_CLOCK_DIV2);
SPI.setBitOrder(MSBFIRST);
SPI.setDataMode(SPI_MODE0);
digitalWrite(CSB1, LOW);
SPI.transfer(0x0f);
SPI.transfer(0x03);
digitalWrite(CSB1, HIGH);//set range +-2g
delayMicroseconds(2);

digitalWrite(CSB1, LOW);
SPI.transfer(0x13);
SPI.transfer(0x00);
digitalWrite(CSB1, HIGH);//set 2000hz
delayMicroseconds(2);

}

int readAcc(unsigned int add) {
int data;
int data_low;
digitalWrite(CSB1, LOW);
SPI.transfer(0x80 + add);
data_low = SPI.transfer(0x00); // read LSB (4 bits)
digitalWrite(CSB1, HIGH);
delayMicroseconds(2);
digitalWrite(CSB1, LOW);
SPI.transfer(0x80 + add + 1); //read MSB (8bits)
 data= (SPI.transfer(0x00) << 4); // sums MSB and LSB
 data+=(data_low >>4);
digitalWrite(CSB1, HIGH);
delayMicroseconds(2);
return data;
}


void loop() {

int x, y, z;
currentMicros = micros();   // capture the latest value of millis()
x = readAcc(x_lsb);
y = readAcc(y_lsb);
z = readAcc(z_lsb);

Serial.print(x);
Serial.print(",");
Serial.print(y);
Serial.print(",");
Serial.print(z);
Serial.print(",");
Serial.println(currentMicros);
}
