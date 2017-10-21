int coil1 = 1;
int coil2 = 2;
int coil3 = 3;
int coil4 = 4;

double reading1;
double reading2;
double reading3;
double reading4;

static char outstr[32];

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  char coil1_tag[] = "coil1";
  char coil2_tag[] = "coil2";
  char coil3_tag[] = "coil3";
  char coil4_tag[] = "coil4";
}

void loop() {
  // put your main code here, to run repeatedly:
  reading1 = analogRead(coil1);
  dtostrf(reading1, 10, 4, outstr);
  strcat(outstr, coil1_tag);
  Serial.println(&outstr);

  reading2 = analogRead(coil2);
  dtostrf(reading2, 10, 4, outstr);
  strcat(outstr, coil2_tag);
  Serial.println(&outstr);
  
  reading3 = analogRead(coil3);
  dtostrf(reading3, 10, 4, outstr);
  strcat(outstr, coil3_tag);
  Serial.println(&outstr);
  
  reading4 = analogRead(coil4);
  dtostrf(reading4, 10, 4, outstr);
  strcat(outstr, coil4_tag);
  Serial.println(&outstr);
  
  //delay(10);
}
