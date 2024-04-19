/******************************************************************************

                            relokate.c
                      Kwang Seob Kim 6/2020
                      Edited by Jessica Gaines 7/2020
                      
*******************************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#define pi 3.14159265


double norm (double ax, double ay, double bx, double by) {
  return sqrt(pow(ax-bx,2) +pow(ay-by,2));
}

void res2center(double *Res_Tx, int n1, double *Res_Ty, int n2, double *ErrMin, int n3, double *Px, int PSize, double *Py, int PSize2, double *Tx, int TSize, double *Ty, int TSize2) {
  printf("Inside res2center");
  double Pmin = 10.0;
  double err = 10.0;
  double angle[50000] = {0.0};
  int i;
  int j;
  int k;
  int newind = 0;
  double bin[1401][3] = {0.0};
  for(i = 0; i <= 1401; i++){
    bin[i][1] = 5.0;
    bin[i][2] = 50; 
  }
  for (k = 0; k < TSize; k++) {
    angle[k] = -atan(Ty[k]/Tx[k]) * (180.0/pi);
    if (Tx[k] > 0) {
      angle[k] =  angle[k] +180.0; 
    }

    if (angle[k] >= 39.95   && angle[k] <= 180.04) {
      newind = (int) round(angle[k]*10-400.0); //if angle is 45.12, it will be 51th index
      // if angle is 45.16 it will be 52nd index
      err = fabs((angle[k]*10-400.0) - (newind));// -51
      Pmin = norm(Px[newind*10],Py[newind*10],Tx[k],Ty[k]);

      if (err < bin[newind][1] && Pmin < bin[newind][2]) {     
        bin[newind][0] = k;
        bin[newind][1] = err;
        bin[newind][2] = Pmin;
      }
    }
  }
  printf("%i", TSize);
  //printf("%i ",c);

  //printf("%f %f %f\n", bin[0][0],bin[0][1],bin[0][2]);
  //printf("%f %f %f\n", bin[400][0],bin[400][1],bin[400][2]);
  int rightind = 0;
  for (j = 0; j < n1; j++) {
    rightind = (int)bin[j][0];
    if (bin[j][1] > 1) {
      printf(" Check3       ");
      printf("%f %f %f %f\n", bin[j][0],bin[j][1],bin[j][2]);
      rightind = rightind +1;
    }
      
    Res_Tx[j] = Tx[rightind];
    Res_Ty[j] = Ty[rightind];
  }
  
  //free(switchjump);
}