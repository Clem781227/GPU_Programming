/**************************************************************
Nested Monte Carlo for bullet option Pricing 
Project By : Afrit Anouar & Clément Gueneau 

Based on code of the original author : Lokman A. Abbas-Turki
***************************************************************/
#include "rng.h"


// Generate uniformly distributed random variables
__device__ void CMRG_d(int *a0, int *a1, int *a2, int *a3, int *a4, 
			     int *a5, float *g0, float *g1, int nb){

 const int m1 = 2147483647;// Requested for the simulation
 const int m2 = 2145483479;// Requested for the simulation
 int h, p12, p13, p21, p23, k, loc;// Requested local parameters

 for(k=0; k<nb; k++){
	 // First Component 
	 h = *a0/q13; 
	 p13 = a13*(h*q13-*a0)-h*r13;
	 h = *a1/q12; 
	 p12 = a12*(*a1-h*q12)-h*r12;

	 if (p13 < 0) {
	   p13 = p13 + m1;
	 }
	 if (p12 < 0) {
	   p12 = p12 + m1;
	 }
	 *a0 = *a1;
	 *a1 = *a2;
	 if( (p12 - p13) < 0){
	   *a2 = p12 - p13 + m1;  
	 } else {
	   *a2 = p12 - p13;
	 }
  
	 // Second Component 
	 h = *a3/q23; 
	 p23 = a23*(h*q23-*a3)-h*r23;
	 h = *a5/q21; 
	 p21 = a21*(*a5-h*q21)-h*r21;

	 if (p23 < 0){
	   p23 = p23 + m2;
	 }
	 if (p12 < 0){
	   p21 = p21 + m2;
	 }
	 *a3 = *a4;
	 *a4 = *a5;
	 if ( (p21 - p23) < 0) {
	   *a5 = p21 - p23 + m2;  
	 } else {
	   *a5 = p21 - p23;
	 }

	 // Combines the two MRGs
	 if(*a2 < *a5){
		loc = *a2 - *a5 + m1;
	 }else{loc = *a2 - *a5;} 

	 if(k){
		if(loc == 0){
			*g1 = Invmp*m1;
		}else{*g1 = Invmp*loc;}
	 }else{
		*g1 = 0.0f; 
		if(loc == 0){
			*g0 = Invmp*m1;
		}else{*g0 = Invmp*loc;}
	 }
  }
}

// Genrates Gaussian distribution from a uniform one (Box-Muller)
__device__ void BoxMuller_d(float *g0, float *g1){

  float loc;
  if (*g1 < 1.45e-6f){
    loc = sqrtf(-2.0f*logf(0.00001f))*cosf(*g0*2.0f*MoPI);
  } else {
    if (*g1 > 0.99999f){
      loc = 0.0f;
    } else {loc = sqrtf(-2.0f*logf(*g1))*cosf(*g0*2.0f*MoPI);}
  }
  *g0 = loc;
}

// Black & Scholes model
__device__ void BS_d(float *S2, float S1, float r0,
					 float sigma, float dt, float e){

  *S2 = S1*expf((r0-0.5f*sigma*sigma)*dt*dt + sigma*dt*e);
}
__device__ void CMRG_get_d(int *a0,int * a1,int * a2,int * a3,int * a4,int * a5,int * CMRG_in){
	
	*a0 = CMRG_in[0];
	*a1 = CMRG_in[1];
	*a2 = CMRG_in[2];
	*a3 = CMRG_in[3];
	*a4 = CMRG_in[4];
	*a5 = CMRG_in[5];

}

__device__ void CMRG_set_d(int *a0,int * a1,int * a2,int * a3,int * a4,int * a5,int * CMRG_in){
	
	CMRG_in[0] = *a0  ;
	CMRG_in[1] = *a1;
	CMRG_in[2] = *a2;
	CMRG_in[3] = *a3;
	CMRG_in[4] = *a4;
	CMRG_in[5] = *a5;

}

__global__ void MC_inner_trajectory_k(float x_0, float r0,
					 float sigma, float dt, int P1, int P2,
					 float K, float *R1, int I_t, float t,
					 float B, int size, int M,
					 TabSeedCMRG_t *pt_cmrg){
	
   int idx = threadIdx.x + blockIdx.x*blockDim.x;
   int a0, a1, a2, a3, a4, a5, pR;
   float U1, U2;
   extern __shared__ float A[];
   float *sSh, *r1SH ;
   sSh = A;
   r1SH = sSh + 2*blockDim.x;

   CMRG_get_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][idx]);
  
   sSh[threadIdx.x] = x_0;
   sSh[threadIdx.x + blockDim.x] = x_0;

   pR = I_t ; 
   
   for (int k=int(t*M); k<=M; k++){
		
       CMRG_d(&a0, &a1, &a2, &a3, &a4, &a5, &U1, &U2, 2);

	   BoxMuller_d(&U1, &U2);
	
	   BS_d(sSh + threadIdx.x + (k%2) * blockDim.x, sSh[threadIdx.x + (k+1)%2 * blockDim.x],r0,sigma, dt, U1)	;

	   pR += (sSh[threadIdx.x+(k%2)*blockDim.x]<B);

   }

    CMRG_set_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][idx]) ;
	r1SH[threadIdx.x] = expf(-r0*dt*dt*M*(1-t))*fmaxf(0.0f, sSh[threadIdx.x+(M%2)*blockDim.x]-K)*((pR<=P2)&&(pR>=P1))/size;
	

	__syncthreads();
	int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
				r1SH[threadIdx.x] += r1SH[threadIdx.x + i] ;
				
		}
		__syncthreads();
		i /= 2 ;
	}
	if(threadIdx.x ==0){
		atomicAdd(R1, r1SH[0]);
	}

}



		
__global__ void MC_k(float *Stock_tracker,float x_0, float r0,
					 float sigma, float dt, int P1, int P2,
					 float K, float *R1, float *R2,
					 int *I_tTracker, float* time_tracker,
					 float B, int size, int size_inner,int M,
					 TabSeedCMRG_t *pt_cmrg){

   int idx = threadIdx.x + blockIdx.x*blockDim.x;
   int a0, a1, a2, a3, a4, a5, pR;
   float U1, U2;
   extern __shared__ float A[];
   float *sSh, *r1SH, *r2SH ;
   sSh = A;
   r1SH = sSh + 2*blockDim.x;
   r2SH = r1SH + blockDim.x;

   CMRG_get_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][idx]);

 
   sSh[threadIdx.x] = x_0;
   
   pR = 0 ; 

   for (int k=1; k<=M; k++){
	
       CMRG_d(&a0, &a1, &a2, &a3, &a4, &a5, &U1, &U2, 2);


	   BoxMuller_d(&U1, &U2);
	
	   BS_d(sSh + threadIdx.x + (k%2) * blockDim.x, sSh[threadIdx.x + (k+1)%2 * blockDim.x],r0,sigma, dt, U1)	;
		


	   pR += (sSh[threadIdx.x+(k%2)*blockDim.x]<B);
	   

	   Stock_tracker[k+ M*idx-1] = sSh[threadIdx.x + (k%2) * blockDim.x];
	   I_tTracker[k+ M*idx-1] = pR;
	   time_tracker[k + M * idx -1] = k*dt*dt;
   }

    CMRG_set_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][idx]) ;
  
	r1SH[threadIdx.x] = expf(-r0*dt*dt*M)*fmaxf(0.0f, sSh[threadIdx.x+(M%2)*blockDim.x]-K)*((pR<=P2)&&(pR>=P1))/size;
	r2SH[threadIdx.x] = r1SH[threadIdx.x]*r1SH[threadIdx.x] * size;

	__syncthreads();
	int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
				r1SH[threadIdx.x] += r1SH[threadIdx.x + i] ;
				r2SH[threadIdx.x] += r2SH[threadIdx.x + i] ;
		}
		__syncthreads();
		i /= 2 ;
	}
	if(threadIdx.x ==0){
		atomicAdd(R1, r1SH[0]);
		atomicAdd(R2, r2SH[0]);
	}

}


int main()
{

	float T = 1.0f;
	float K = 100.0f;
	float x_0 = 100.0f;
	float vol = 0.2f;
	float tau = 0.1f;
	float B = 120.0f;
	int M = 100;
	int P1 = 10;
	int P2 = 49;
	float dt = sqrtf(T/M);
	float sum = 0.0f;	
	float sum2 = 0.0f;
	float Tim;							// GPU timer instructions
	cudaEvent_t start, stop;			// GPU timer instructions
	float *res1, *res2;

	int Ntraj =64 * 64;
	int Ntraj_inner = 64 * 64 ;
	
	float *Stock_tracker;				//Array to save stock values of inner trajectories
	int *I_tTracker;					//Array to save I_t values of inner trajectories
	float *time_tracker;                //Array to save times of inner trajectories
	float *inner_price_resultGPU ;		//Array to save price estimations in inner trajectories
	float *inner_price_resultCPU  ;   

	cudaMalloc(&time_tracker, Ntraj*M*sizeof(float));
	cudaMalloc(&Stock_tracker, Ntraj*M*sizeof(float));
	cudaMalloc(&I_tTracker, Ntraj*M*sizeof(int));
	
	cudaMalloc(&inner_price_resultGPU,Ntraj*M*sizeof(float));
	cudaMemset(inner_price_resultGPU, 0,Ntraj*M* sizeof(float));
	inner_price_resultCPU = (float*)malloc(Ntraj * M *sizeof(float));

	
	cudaMalloc(&res1, sizeof(float));
	cudaMalloc(&res2, sizeof(float));
	
	
	cudaMemset(res1, 0, sizeof(float));
	cudaMemset(res2, 0, sizeof(float)); 



	float *S_t, *t;						// CPU equivalents of Stock_tracker, I_tTracker, time_tracker
	int *I_t ; 
	
	S_t = (float*)malloc(Ntraj*M*sizeof(float));
	t = (float*)malloc(Ntraj*M*sizeof(float));
	I_t = (int*)malloc(Ntraj*M*sizeof(int));


	PostInitDataCMRG();

	cudaEventCreate(&start);			// GPU timer instructions
	cudaEventCreate(&stop);				// GPU timer instructions
	cudaEventRecord(start,0);			// GPU timer instructions


	//Generate Outer trajectories and values of (Time, Stock, I_t) to be used later of inner trajectories
	MC_k<<<64,64,4*64*sizeof(float)>>>(Stock_tracker,x_0, tau, vol, dt, 
				  P1, P2, K, res1, res2,I_tTracker , time_tracker,
				  B, Ntraj, Ntraj_inner, M, CMRG);
	

	
	//Transfer the data (Time, Stock, I_t) into CPU 
	cudaMemcpy(S_t, Stock_tracker, Ntraj*M * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(t, time_tracker, Ntraj*M * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(I_t, I_tTracker, Ntraj*M * sizeof(int), cudaMemcpyDeviceToHost);

	cudaMemcpy(&sum, res1, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&sum2, res2, sizeof(float), cudaMemcpyDeviceToHost);

	//Simulation of inner trajectories, each one starting for a set of (Time, Stock, I_t) 
	// Generated by the outer trajectories kernel. 
	for(int i = 0; i< Ntraj * M ; i++){
		MC_inner_trajectory_k<<<8,512,3*512*sizeof(float)>>>(S_t[i],tau,
					 vol, dt, P1, P2,
					 K, inner_price_resultGPU + i,I_t[i], t[i],
					  B, Ntraj_inner, M,
					 CMRG);
		if (i % Ntraj == 0) printf(" Progress %i%%/%i%% \n",i/Ntraj,M);
	}
	//Wait for all Kernels to finish filling their results in inner_price_resultGPU. 
	cudaDeviceSynchronize();

	cudaEventRecord(stop,0);			// GPU timer instructions
	cudaEventSynchronize(stop);			// GPU timer instructions
	cudaEventElapsedTime(&Tim,			// GPU timer instructions
			 start, stop);				// GPU timer instructions
	cudaEventDestroy(start);			// GPU timer instructions
	cudaEventDestroy(stop);				// GPU timer instructions

	// Transfer Nested price results to a CPU Array "inner_price_resultCPU"
	cudaMemcpy(inner_price_resultCPU, inner_price_resultGPU,Ntraj * M * sizeof(float), cudaMemcpyDeviceToHost);
	
	
	//Printing a Sample of the data generated 
	printf("###########################\n");
	printf("Data Sample\n");
	for(int k = 0;k< 15*M;k++){
		if (k%100 == 0)printf("#######################\n");
		printf("time = %f Stock = %f  I_t = %i Price = %f \n",t[k],S_t[k],I_t[k],inner_price_resultCPU[k] );
	}
	printf("###########################\n");


	//Writing the data generated into a ".csv" file
	float a,c,d ;
	int b ;
	FILE *outfile =fopen("Nested_MC_Bullet_Option.csv","w");
	fprintf(outfile,"Time,Stock,I_t,Price\n");
	for (int k = 0;k < Ntraj*M; ++k){
		a = S_t[k];
		b = I_t[k];
		c = t[k];
		d = inner_price_resultCPU[k];

		fprintf(outfile,"%f,%f,%i,%f\n",c,a,b,d);
	}
	fclose(outfile);

	//Freeing all memeory used
	cudaFree(res1);
	cudaFree(res2);
	cudaFree(time_tracker);
	cudaFree(Stock_tracker);
	cudaFree(I_tTracker);
	cudaFree(inner_price_resultGPU) ; 
	free(inner_price_resultCPU);

	

	// Printing the price of the option at time 0
	printf("The price at time zero is equal to %f\n", sum);
	printf("error associated to a confidence interval of 95%% = %f\n", 
		   1.96*sqrt((double)(1.0f/(Ntraj-1))*(Ntraj*sum2 - (sum*sum)))/
		   sqrt((double)Ntraj));

	//Printing Execution time for outer trajectories and inner trajectories generation
	printf("Execution time for %i terminal Payoffs is %f ms\n", Ntraj * M * Ntraj_inner,Tim);

	FreeCMRG();
	
	return 0;
}


