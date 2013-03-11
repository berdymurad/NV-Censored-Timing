//============================================================================
// Name        : NVTiming-T4E.cpp
// Author      : Tong
// Version     :
// Copyright   : ...
// Description : general code for newsvendor with censored demand --- the event case (up to 4 periods)
//============================================================================

//***********************************************************

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#include <vector>
using std::vector;


//***********************************************************

#define N 4						//# of periods
#define LAMBDA_STEP 1000
#define X_MAX 120 //400
#define D_MAX 600 //1000

#define D_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN 8
#define LAMBDA_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN 8
#define LAMBDA_LOW_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN 4

#define X_MYOPIC_UP_MULTIPLE_OF_MEAN 10
//***********************************************************

double price, cost;
double alpha0, beta0;

double demand_pdf2c[D_MAX+1][X_MAX+1];	//for pdf with 1 censored observation
double demand_pdf3ec[D_MAX+1][D_MAX+1][X_MAX+1]; //for pdf with 1 exact and 1 censored observations
double demand_pdf3cc[D_MAX+1][X_MAX+1][X_MAX+1]; //for pdf with 2 censored observations (the two observations are ordered decreasingly!!)
double demand_pdf4eec[D_MAX+1][2*D_MAX+1][X_MAX+1]; //for pdf with 2 exact and 1 censored observations
double demand_pdf4ecc[D_MAX+1][D_MAX+1][X_MAX+1][X_MAX+1]; //for pdf with 1 exact and 2 censored observations
double demand_pdf4ccc[D_MAX+1][X_MAX+1][X_MAX+1][X_MAX+1]; //for pdf with 3 censored observations (the two observations are ordered decreasingly!!)


FILE *fp;									//output files

//***********************************************************

int int_max(int x, int y)
{
	if (x >= y) return x;
	else return y;
}

int int_min(int x, int y)
{
	if (x >= y) return y;
	else return x;
}

int int_max3(int x, int y, int z)
{
	if (x >= y) return int_max(x,z);
	else return int_max(y,z);
}

int int_mid3(int x, int y, int z)
{
	if (x >= y)
	{
		if (z>=x) return x;
		else if (z>=y) return z;
		else return y;
	}
	else
	{
		if (z>=y) return y;
		else if (z>=x) return z;
		else return x;
	}

}

int int_min3(int x, int y, int z)
{
	if (x >= y) return int_min(y,z);
	else return int_min(x,z);
}


//***********************************************************

double Poisson(int xx, double lam)
{
	double log_pmf = xx*log(lam) -lam - lgamma(xx+1);

	return exp(log_pmf);
}

double Gamma(double xx, double aa, double bb)
{
	double log_pdf = 0;

	if ((xx>=0)&&(aa>0)&&(bb>0))
		log_pdf = aa*log(bb) + log(xx)*(aa-1)  -bb*xx - lgamma(aa);

	return exp(log_pdf);
}

double NegBinomial(int kk, double rr, double pp)
{
	double log_pmf = 0;

	if ((kk>=0)&&(rr>0)&&(pp>=0)&&(pp<=1))
		log_pmf = lgamma(kk+rr) - lgamma(kk+1) - lgamma(rr)  + log(1-pp)*rr + log(pp)*kk;

	return exp(log_pmf);
}

//***********************************************************

//first-order difference of L := L(x+1) - L(x)
double L_prime(int x, double alpha, double beta)
{

	double r = alpha;
	double p = 1/(1+beta);

	double Phi_x = 0;

	for (int d=0; d<=x; d++)
		Phi_x += NegBinomial(d, r, p);

	double Phi_bar_x = 1 - Phi_x;

	return price * Phi_bar_x - cost;

}

//search for myopic inventory level with updated knowledge (alpha, beta) (only for cases without censorship!!)
int find_x_myopic(double alpha, double beta)
{
	//bi-sectional search for x such that L_prime is zero

	int x;
	int x_up = (int) X_MYOPIC_UP_MULTIPLE_OF_MEAN * alpha / beta;
	int x_low = 0;

	while (x_up - x_low > 3)
	{
		x = (x_up + x_low)/2;

		double temp = L_prime(x, alpha, beta);

		if (temp > 0)
			x_low = x+1;
		else
			x_up = x;
	}

	for (x=x_low; x<=x_up; x++)
	{
		double temp = L_prime(x, alpha, beta);

		if (temp < 0)
			break;
	}

	return x;

}



//***********************************************************

//calculate the likelihood of historical censored observations ($numberOfStockout$, $x_observed$) with a given lambda
double F_bar(double lambda, vector<int> x_observed)
{
	double out=1;

	for (int j = 0; j<x_observed.size(); j++)
	{
		double F_bar_j = 0;
		for (int d=0; d<x_observed[j]; d++)
			F_bar_j += Poisson(d, lambda);
		F_bar_j = fmax(0, 1 - F_bar_j);

		out *= F_bar_j;
	}

	return out;
}

//bayesian updating of pdf function $pdf$ based on historical censored observations ($n$, $d_observed$, $x_observed$)
void pdf_update(vector<double>& pdf, int n, int d_observed, vector<int> x_observed)
{
	int numberOfStockout = x_observed.size();

	//update alpha,beta for $n-numberOfStockout$ exact observations $d_observed$
	double alpha_n = alpha0 + d_observed;
	double beta_n = beta0 + n-1-numberOfStockout;

	double r = alpha_n;
	double p = 1/(1+beta_n);

	double d_mean = r*p/(1-p);
	double d_stdev = sqrt(r*p)/(1-p);
	int d_up = fmin(D_MAX, d_mean + D_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN * d_stdev);  //upper bound: mean + 5*var

	double lambda_mean = alpha_n/beta_n;
	double lambda_stdev = sqrt(alpha_n)/beta_n;
	double lambda_up = lambda_mean + LAMBDA_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN*lambda_stdev;
	double lambda_low = fmax(0, lambda_mean - LAMBDA_LOW_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN*lambda_stdev);
	double delta_lambda = (lambda_up-lambda_low) / LAMBDA_STEP;

	//initialize pdf for lambda
	double * lambda_pdf = new double[LAMBDA_STEP+1];

	//calculate the denominator in Baysian equation (predictive distribution)
	double denominator=0;
	for (int i=1; i<=LAMBDA_STEP; i++)
	{
		double lambda = lambda_low+i* delta_lambda;
		denominator += F_bar(lambda, x_observed) * Gamma(lambda, alpha_n, beta_n) * delta_lambda;
	}

	//calculate the Bayesian posterior for lambda
	for (int i=1; i<=LAMBDA_STEP; i++)
	{
		double lambda = lambda_low+i* delta_lambda;
		lambda_pdf[i] = F_bar(lambda, x_observed) * Gamma(lambda, alpha_n, beta_n) / denominator ;
	}

	//calculate predictive demand distribution with the updated posterior
	for (int d=0; d<=d_up; d++)
	{
		double intg = 0;
		for (int i=1; i<=LAMBDA_STEP; i++)
			intg += lambda_pdf[i] * Poisson(d, lambda_low+i* delta_lambda) * delta_lambda;

		pdf[d] = intg;

		//save the calculated dist in array
		if ((n==2)&&(x_observed[0]<=X_MAX))
			demand_pdf2c[d][x_observed[0]] = intg;
		else if ((n==3)&&(numberOfStockout==1)&&(x_observed[0]<=X_MAX))
			demand_pdf3ec[d][d_observed][x_observed[0]] = intg;
		else if ((n==3)&&(numberOfStockout==2)&&(int_max(x_observed[0],x_observed[1])<=X_MAX))
			demand_pdf3cc[d][int_max(x_observed[0],x_observed[1])][int_min(x_observed[0],x_observed[1])] = intg;
		else if ((n==4)&&(numberOfStockout==1)&&(x_observed[0]<=X_MAX))
			demand_pdf4eec[d][d_observed][x_observed[0]] = intg;
		else if ((n==4)&&(numberOfStockout==2)&&(int_max(x_observed[0],x_observed[1])<=X_MAX))
			demand_pdf4ecc[d][d_observed][int_max(x_observed[0],x_observed[1])][int_min(x_observed[0],x_observed[1])] = intg;
		else if ((n==4)&&(numberOfStockout==3)&&(int_max3(x_observed[0],x_observed[1],x_observed[2])<=X_MAX))
			demand_pdf4ccc[d][int_max3(x_observed[0],x_observed[1],x_observed[2])][int_mid3(x_observed[0],x_observed[1],x_observed[2])][int_min3(x_observed[0],x_observed[1],x_observed[2])] = intg;
	}

	delete [] lambda_pdf;

}

//load or calculate updated pdf
void pdf_censored(vector<double>& pdf, int n, int d_observed, vector<int> x_observed)
{
	int numberOfStockout = x_observed.size();

	//try to load from array if the pdf has been calculated before
	if ((n==2)&&(x_observed[0]<=X_MAX)&&(demand_pdf2c[0][x_observed[0]] > -1))
	{
		for (int i=0; i<pdf.size(); i++)
			pdf[i] = demand_pdf2c[i][x_observed[0]];
	}
	else if ((n==3)&&(numberOfStockout==1)&&(x_observed[0]<=X_MAX)&&(demand_pdf3ec[0][d_observed][x_observed[0]] > -1))
	{
		for (int i=0; i<pdf.size(); i++)
			pdf[i] = demand_pdf3ec[i][d_observed][x_observed[0]];
	}
	else if ((n==3)&&(numberOfStockout==2)&&(int_max(x_observed[0],x_observed[1])<=X_MAX)&&(demand_pdf3cc[0][int_max(x_observed[0],x_observed[1])][int_min(x_observed[0],x_observed[1])] > -1))
	{
		for (int i=0; i<pdf.size(); i++)
			pdf[i] = demand_pdf3cc[i][int_max(x_observed[0],x_observed[1])][int_min(x_observed[0],x_observed[1])];
	}
	else if ((n==4)&&(numberOfStockout==1)&&(x_observed[0]<=X_MAX)&&(demand_pdf4eec[0][d_observed][x_observed[0]] > -1))
	{
		for (int i=0; i<pdf.size(); i++)
			pdf[i] = demand_pdf4eec[i][d_observed][x_observed[0]];
	}
	else if ((n==4)&&(numberOfStockout==2)&&(int_max(x_observed[0],x_observed[1])<=X_MAX)&&(demand_pdf4ecc[0][d_observed][int_max(x_observed[0],x_observed[1])][int_min(x_observed[0],x_observed[1])] > -1))
	{
		for (int i=0; i<pdf.size(); i++)
			pdf[i] = demand_pdf4ecc[i][d_observed][int_max(x_observed[0],x_observed[1])][int_min(x_observed[0],x_observed[1])];
	}
	else if ((n==4)&&(numberOfStockout==3)&&(int_max3(x_observed[0],x_observed[1],x_observed[2])<=X_MAX)&&(demand_pdf4ccc[0][int_max3(x_observed[0],x_observed[1],x_observed[2])][int_mid3(x_observed[0],x_observed[1],x_observed[2])][int_min3(x_observed[0],x_observed[1],x_observed[2])] > -1))
	{
		for (int i=0; i<pdf.size(); i++)
			pdf[i] = demand_pdf4ccc[i][int_max3(x_observed[0],x_observed[1],x_observed[2])][int_mid3(x_observed[0],x_observed[1],x_observed[2])][int_min3(x_observed[0],x_observed[1],x_observed[2])];
	}
	else //otherwise calculate by calling pdf_update()
		pdf_update(pdf, n, d_observed, x_observed);

}



//Dynamic Program recursion
double G_COE(int n, int x, int d_observed, vector<int> x_observed);
double V_COE(int n, int d_observed, vector<int> x_observed)
{
	if (n==N+1) return 0;	//V_{N+1}()=0
	else
	{

		//search for optimal inventory level x
		int x = 0;

		//initial lower bound of x is set to 0
		int x_low = 0;

		//if no censorship before, use myopic inventory level as lower bound
		if (x_observed.size()==0)
			x_low = find_x_myopic(alpha0 + d_observed, beta0+n-1);

		int x_opt = x_low;

		//evaluate low bound x_low
		double v_max = G_COE(n, x_low, d_observed, x_observed );

		//linear search from x_low onwards
		for (x=x_low+1;;x++)
		{
			double temp = G_COE(n, x, d_observed, x_observed );

			if (temp > v_max)
			{
				x_opt = x;
				v_max = temp;
			}
			else break;
		}

		if (n==1)
		{
			printf("%d\t%f\t", x_opt, v_max);
			fprintf(fp, "%d\t%f\t", x_opt, v_max);
		}

		return v_max;

	}
}



double G_COE(int n, int x, int d_observed, vector<int> x_observed)
{
	//update alpha,beta for $n-numberOfStockout$ exact observations $d_observed$
	double alpha_n = alpha0 + d_observed;
	double beta_n = beta0 + n-1-x_observed.size();

	double r = alpha_n;
	double p = 1/(1+beta_n);


	if (x_observed.size() == 0)	//if no censorship happened before, use the updating mechanism with exact observations (predictive demand is negative binomial)
	{
		double out1 = 0;

		//no stockout
		//#pragma omp parallel for schedule(dynamic) reduction(+:out1)
		for (int d=0; d<=x-1; d++)
		{
			int d_o = d_observed + d;
			out1 += ( price*d + V_COE(n+1, d_o, x_observed) ) * NegBinomial(d, r, p);
		}

		//with stockout
		vector<int> x_o = x_observed;
		x_o.push_back(x);

		double Phi_x = 0;
		for (int d=0; d<=x-1; d++)
			Phi_x += NegBinomial(d, r, p);
		double F_bar_x = fmax(0,1-Phi_x);

		out1 += (price*x + V_COE(n+1, d_observed, x_o)) * F_bar_x;

		return out1 - cost*x;
	}
	else	//if there is censorship, use the updating mechanism coded in pdf_censored()
	{
		double out2 = 0;

		double d_mean = r*p/(1-p);
		double d_stdev = sqrt(r*p)/(1-p);
		int d_up = fmin(D_MAX, d_mean + D_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN * d_stdev);  //upper bound: mean + 5*var

		vector<double> demand_pdf (d_up+1);
		pdf_censored(demand_pdf, n, d_observed, x_observed);

		//no stockout
		#pragma omp parallel for schedule(dynamic) reduction(+:out2)
		for (int d=0; d<=x-1; d++)
		{
			int d_o = d_observed + d;
			out2 += ( price*d + V_COE(n+1, d_o, x_observed) ) * demand_pdf[d];
		}

		//with stockout
		vector<int> x_o = x_observed;
		x_o.push_back(x);

		double Phi_x = 0;
		for (int d=0; d<=x-1; d++)
			Phi_x += demand_pdf[d];
		double F_bar_x = fmax(0,1-Phi_x);

		out2 += ( price*x + V_COE(n+1, d_observed, x_o) ) * F_bar_x;

		return out2 - cost*x;
	}



}



int main(void)
{
	//omp_set_num_threads(omp_get_num_procs());
	printf("Num of Procs: %d\n",omp_get_num_procs());
	printf("Num of Threads: %d\n",omp_get_num_threads());


	//Open output file
	if ((fp=fopen("NVTiming-T4E50c.txt","w"))==NULL)
	{
		printf("%s","Open file \"NVTiming-T4E.txt\" error!!");
		return -1;
	}

	fprintf(fp, "r\tc\talpha\tbeta\tQ_Event\tPi_Event\tTime\n");




	//initialize cost parameters
	price = 2;
	cost = 0.1;

	//initialize info parameters
	int lambda_mean = 50;
	beta0 = 0.0625;

	//for (lambda_mean=10; lambda_mean<=50; lambda_mean+=10)
	//for (lambda_mean=75; lambda_mean<=100; lambda_mean+=25)
	for (beta0=0.0625; beta0<=2.5; beta0*=2)
	//for (cost=1.9;cost>0.15;cost-=0.1)
	{
		alpha0 = beta0*lambda_mean;




		//intialize the arrays for storing calculated pdf's
		#pragma omp parallel for schedule(static)
		for (int i=0;i<=X_MAX;i++)
		{
			demand_pdf2c[0][i] = -8;
			for (int j=1;j<=D_MAX;j++)
				demand_pdf2c[j][i] = 0;
		}


		#pragma omp parallel for schedule(static)
		for (int i=0;i<=X_MAX;i++)
		for (int j=0;j<=D_MAX;j++)
		{
			demand_pdf3ec[0][j][i] = -8;

			for (int k=1;k<=D_MAX;k++)
				demand_pdf3ec[k][j][i] = 0;
		}

		#pragma omp parallel for schedule(static)
		for (int i=0;i<=X_MAX;i++)
		for (int j=0;j<=X_MAX;j++)
		{
			demand_pdf3cc[0][j][i] = -8;
			for (int k=1;k<=D_MAX;k++)
				demand_pdf3cc[k][j][i] = 0;
		}

		#pragma omp parallel for schedule(static)
		for (int i=0;i<=X_MAX;i++)
		for (int j=0;j<=2*D_MAX;j++)
		{
			demand_pdf4eec[0][j][i] = -8;
			for (int l=1;l<=D_MAX;l++)
				demand_pdf4eec[l][j][i] = 0;
		}

		#pragma omp parallel for schedule(static)
		for (int i=0;i<=X_MAX;i++)
		for (int j=0;j<=X_MAX;j++)
		for (int k=0;k<=D_MAX;k++)
		{
			demand_pdf4ecc[0][k][j][i] = -8;
			for (int l=1;l<=D_MAX;l++)
				demand_pdf4ecc[l][k][j][i] = 0;
		}

		#pragma omp parallel for schedule(static)
		for (int i=0;i<=X_MAX;i++)
		for (int j=0;j<=X_MAX;j++)
		for (int k=0;k<=X_MAX;k++)
		{
			demand_pdf4ccc[0][k][j][i] = -8;
			for (int l=1;l<=D_MAX;l++)
				demand_pdf4ccc[l][k][j][i] = 0;
		}


		printf("%.1f\t%.2f\t%.4f\t%.4f\t", price, cost, alpha0, beta0);
		fprintf(fp, "%.1f\t%.2f\t%.4f\t%.4f\t", price, cost, alpha0, beta0);

		//initial observations are null
		vector<int> x_observed;


		double startTime = omp_get_wtime();
		V_COE(1, 0, x_observed);
		double endTime = omp_get_wtime();


		printf("%f\n", endTime - startTime);
		fprintf(fp, "%f\n", endTime - startTime);

		fflush(fp);

	}

	fclose(fp);
	return 0;
}

