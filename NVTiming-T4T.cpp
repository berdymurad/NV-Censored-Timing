//============================================================================
// Name        : NVTiming-T4.cpp
// Author      : Tong
// Version     :
// Copyright   : ...
// Description : Hello World in C++, Ansi-style
//============================================================================

//***********************************************************

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

//***********************************************************

#define N 4						//# of periods
#define T_STEP 500
#define LAMBDA_STEP 1000

#define D_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN 10

#define X_MYOPIC_UP_MULTIPLE_OF_MEAN 10
//***********************************************************

double price, cost;
double alpha0, beta0;

FILE *fp;									//output files

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



double InvBeta2(double tt, int xx, double aa, double bb)
{
	if (tt==0) return 0;
	else
	{
		double log_pdf = 0;


		if ((tt>0)&&(xx>0)&&(aa>0)&&(bb>0))
			log_pdf = (xx-1)*log(tt) + aa*log(bb) - (xx+aa) * log(tt+bb) - ( lgamma(xx) + lgamma(aa) - lgamma(xx+aa) ) ;
			//pdf = pow(tt, xx-1) * pow(bb, aa) * pow(tt+bb, -xx-aa) / Beta(xx,aa);

		return exp(log_pdf);
	}
}

//***********************************************************

double L(int x, double alpha, double beta)
{
	if (x==0) return 0;
	else
	{
		double r = alpha;
		double p = 1/(1+beta);

		double sales = 0;
		double Phi_x = 0;
		for (int d=0; d<=x; d++)
		{
			double prob = NegBinomial(d, r, p);
			sales += d * prob;
			Phi_x += prob;
		}

		sales += x * fmax(0, 1-Phi_x);

		return price * sales - cost * x;
	}
}


double L_prime(int x, double alpha, double beta)  //first-order difference of L := L(x+1) - L(x)
{

	double r = alpha;
	double p = 1/(1+beta);

	double Phi_x = 0;

	for (int d=0; d<=x; d++)
		Phi_x += NegBinomial(d, r, p);

	double Phi_bar_x = 1 - Phi_x;

	return price * Phi_bar_x - cost;

}

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



double G_COT(int n, int x, double alpha, double beta);

double V_COT(int n, double alpha, double beta)
{
	int x = 0;
	int x_low = find_x_myopic(alpha, beta);
	int x_opt = x_low;
	double v_max = G_COT(n, x_low, alpha, beta);

	for (x=x_low+1;;x++)
	{
		double temp = G_COT(n, x, alpha, beta);

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

double G_COT(int n, int x, double alpha, double beta)
{
	double out = 0;

	if (n<N)
	{
		if (x==0)
			return V_COT(n+1, alpha, beta);
		else
		{
			//no stockout
			double r = alpha;
			double p = 1/(1+beta);

			//#pragma omp parallel for schedule(dynamic) reduction(+:out)
			for (int d=0; d<=x-1; d++)
				out += V_COT(n+1, alpha+d, beta+1) * NegBinomial(d, r, p);


			//with stockout
			double delta_t = 1.0/T_STEP;

			//Parallelize by OpenMP Directive
			#pragma omp parallel for schedule(dynamic) reduction(+:out)
			for (int i=1;i<=T_STEP;i++)  //t=i*delta_t //for (double t=0; t<1; t+=delta_t)
				out += V_COT(n+1, alpha+x, beta+(i-0.5)*delta_t) * InvBeta2((i-0.5)*delta_t, x, alpha, beta) * delta_t;

		}

	}

	out += L(x, alpha, beta);

	return out;

}



double G_FO(int n, int x, double alpha, double beta);

double V_FO(int n, double alpha, double beta)
{
	int x_opt = find_x_myopic(alpha, beta);

	double v_max = G_FO(n, x_opt, alpha, beta);

	if (n==1)
	{
		printf("%d\t%f\t", x_opt, v_max);
		fprintf(fp, "%d\t%f\t", x_opt, v_max);
	}

	return v_max;
}

double G_FO(int n, int x, double alpha, double beta)
{
	double out = 0;

	if (n<N)
	{
		double r = alpha;
		double p = 1/(1+beta);

		double d_mean = r*p/(1-p);
		double d_var = r*p/pow(1-p,2.0);
		int d_up = d_mean + D_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN*sqrt(d_var);  //upper bound: mean + 5*var

		//Parallelize by OpenMP Directive
		#pragma omp parallel for schedule(dynamic) reduction(+:out)
		for (int d=0; d<=d_up; d++)
			out += (price*fmin(x,d) + V_FO(n+1, alpha+d, beta+1)) * NegBinomial(d, r, p);

		out -= cost*x;

	}
	else
		out = L(x, alpha, beta);

	return out;

}



int main(void)
{
	//omp_set_num_threads(omp_get_num_procs());
	printf("Num of Procs: %d\n",omp_get_num_procs());
	printf("Num of Threads: %d\n",omp_get_num_threads());

	int i,j,k;

	//Open output file
	if ((fp=fopen("NVTiming-T4T.txt","w"))==NULL)
	{
		printf("%s","Open file \"NVTiming-T4.txt\" error!!");
		return -1;
	}

	fprintf(fp, "r\tc\talpha\tbeta\tQ_Timing\tPi_Timing\tQ_FullObs\tPi_FullObs\tTime\n");




	//initialize lead time parameters
	price = 2;
	cost = 1;

	int lambda_mean = 50;
	beta0 = 0.125;

	//for (lambda_mean=10; lambda_mean<=50; lambda_mean+=10)
	//for (lambda_mean=75; lambda_mean<=100; lambda_mean+=25)
	//for (beta0=0.25; beta0<=2.5; beta0*=2)
	for (cost=1.4;cost>1.15;cost-=0.1)
	{
		alpha0 = beta0*lambda_mean;


		printf("%.1f\t%.2f\t%.4f\t%.4f\t", price, cost, alpha0, beta0);
		fprintf(fp, "%.1f\t%.2f\t%.4f\t%.4f\t", price, cost, alpha0, beta0);

		double startTime = omp_get_wtime();
		//V_COE_1(alpha0, beta0);
		V_COT(1, alpha0, beta0);
		//V_FO(1, alpha0, beta0);
		double endTime = omp_get_wtime();

		printf("%f\n", endTime - startTime);
		fprintf(fp, "%f\n", endTime - startTime);

		fflush(fp);

	}

	fclose(fp);
	return 0;
}
