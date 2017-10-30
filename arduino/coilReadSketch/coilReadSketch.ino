int coil1 = 1;
int coil2 = 2;
int coil3 = 3;
int coil4 = 4;

double reading1;
double reading2;
double reading3;
double reading4;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
}

void loop() {
  // put your main code here, to run repeatedly:
  reading1 = analogRead(coil1);
  Serial.print("coil1: ");
  Serial.println(reading1);

  reading2 = analogRead(coil2);
  Serial.print("coil2: ");
  Serial.println(reading2);
  
  reading3 = analogRead(coil3);
  Serial.print("coil3: ");
  Serial.println(reading3);
  
  reading4 = analogRead(coil4);
  Serial.print("coil4: ");
  Serial.println(reading4);
  
  //delay(10);
}
