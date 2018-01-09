void setup()
{
    // put your setup code here, to run once:
    Error = E;
    P Gain = P;
    I Gain = I;
    D Gain = D;

    E = ; //Set Value - Sensor Data
    P = 1;
    I = 0; //for now we just touch only P and D controller.
    D = 0;
}

void loop()
{
    //P controller //
    Output_P = P * E;

    // I controller //
    //Make an array to store old values of E
    //for (int i = 0; i < 10; i++){
    //  SumE = E(i) + E(i+1) + E(i+2) + E(i+3) + E(i+4) + E;
    //  Output_I = I*(SumE/Step);

    //IStep = 10
    //SumE = E(0) + E(1) + ...
    //Output_I = I * (SumE/Step)

    //D controller //
    DStep = 2;
    DelE = E(1) - E(0);
    Output_D = D * (DelE / DStep);

    //Full Output
    Speed = ...; //insert your ideal speed
    Output_Full = Output_P + Output_I + Output_D;
    Motor_1_Output = Speed * Output_Full;
}
