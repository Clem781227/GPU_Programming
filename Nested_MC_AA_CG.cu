/**************************************************************
Nested Monte Carlo for bullet option Pricing 
Project By : Afrit Anouar & Cl�ment Gueneau 

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

__global__ void MC_inner_trajectory_k(float *x_0, float r0,
					 float sigma, float dt, int P1, int P2,
					 float K, float *R1, int *I_t, float *t,
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
   //if(idx == 3) printf("%i , %i, %i, %i, %i, %i \n",a0, a1, a2, a3, a4, a5)	;				
   sSh[threadIdx.x] = x_0[blockIdx.x];
   sSh[threadIdx.x + blockDim.x] = x_0[blockIdx.x];

   pR = I_t[blockIdx.x] ; 
   
   for (int k=int(t[blockIdx.x]*M); k<=M; k++){
		
       CMRG_d(&a0, &a1, &a2, &a3, &a4, &a5, &U1, &U2, 2);
	   BoxMuller_d(&U1, &U2);
	
	   BS_d(sSh + threadIdx.x + (k%2) * blockDim.x, sSh[threadIdx.x + (k+1)%2 * blockDim.x],r0,sigma, dt, U1)	;

	   pR += (sSh[threadIdx.x+(k%2)*blockDim.x]<B);
	   if (pR > P2) break;

   }
   	
	CMRG_set_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][idx]) ;
	if(pR > P2 || pR < P1 ){
		r1SH[threadIdx.x] = 0 ;
	}else{
		r1SH[threadIdx.x] = expf(-r0*dt*dt*M*(1-t[blockIdx.x]))*fmaxf(0.0f, sSh[threadIdx.x+(M%2)*blockDim.x]-K)*((pR<=P2)&&(pR>=P1))/size;
	}
	
	

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
		atomicAdd(R1 + blockIdx.x, r1SH[0]);
	}

}



		
__global__ void MC_k(float *Stock_tracker,float x_0, float r0,
					 float sigma, float dt, int P1, int P2,
					 float K, float *R1, 
					 int *I_tTracker, float* time_tracker,
					 float B, int size, int size_inner,int M,
					 TabSeedCMRG_t *pt_cmrg){

   int idx = threadIdx.x + blockIdx.x*blockDim.x;
   int a0, a1, a2, a3, a4, a5, pR;
   float U1, U2;
   extern __shared__ float A[];
   float *sSh, *r1SH;
   sSh = A;
   r1SH = sSh + 2*blockDim.x;
   

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
	//float sum2 = 0.0f;
	float Tim;							// GPU timer instructions
	cudaEvent_t start, stop;			// GPU timer instructions
	float *res1;

	int Ntraj = 16 * 32 ;
	int Ntraj_inner = 32 * 32 ;
	
	//int Ntraj =4*4;
	//int Ntraj_inner = 4*4 ;
	
	float *Stock_tracker;				//Array to save stock values of inner trajectories
	int *I_tTracker;					//Array to save I_t values of inner trajectories
	float *time_tracker;                //Array to save times of inner trajectories
	float *inner_price_resultGPU ;		//Array to save price estimations in inner trajectories
	float *inner_price_resultCPU  ;   

	cudaMalloc(&time_tracker, Ntraj*M*sizeof(float));
	cudaMalloc(&Stock_tracker, Ntraj*M*sizeof(float));
	
	cudaMalloc(&I_tTracker, Ntraj*M*sizeof(int));
	//cudaMemset(I_tTracker, 0, Ntraj*M*sizeof(int));

	cudaMalloc(&inner_price_resultGPU,Ntraj*M*sizeof(float));
	cudaMemset(inner_price_resultGPU, 0,Ntraj*M* sizeof(float));
	inner_price_resultCPU = (float*)malloc(Ntraj * M *sizeof(float));

	
	cudaMalloc(&res1, sizeof(float));
	//cudaMalloc(&res2, sizeof(float));
	
	
	cudaMemset(res1, 0, sizeof(float));
	//cudaMemset(res2, 0, sizeof(float)); 



	float *S_t, *t;						// CPU equivalents of Stock_tracker, I_tTracker, time_tracker
	int *I_t ; 
	
	S_t = (float*)malloc(Ntraj*M*sizeof(float));
	t = (float*)malloc(Ntraj*M*sizeof(float));
	I_t = (int*)malloc(Ntraj*M*sizeof(int));



	printf("################### Nested Monte Carlo for a Bullet Option ####################\n");
	printf("###############################################################################\n");
	printf("Prameters :\n Stock = %f\n Strike = %f\n volatility = %f\n riskless rate = %F\n Barrier = %f\n upper limit = %i\n lower limit = %i\n", x_0, K, vol, tau, B, P2, P1);
	printf("Simulation caracteristics :\n Number of outer trajectories = %i\n Number of inner trajectories = %i\n Number of discretisations = %i\n Total number of trajectories = %i\n"
																		,Ntraj, Ntraj_inner, M, Ntraj_inner * Ntraj * M);

	printf("Generating random state ...\n");
	PostInitDataCMRG();
	printf("Random state generation successful.\n");
	cudaEventCreate(&start);			// GPU timer instructions
	cudaEventCreate(&stop);				// GPU timer instructions
	cudaEventRecord(start,0);			// GPU timer instructions

	printf("Generating %i outer trajectories and %i X %i X %i total trajectories...\n", Ntraj, M, Ntraj, Ntraj_inner);
	//Generate Outer trajectories and values of (Time, Stock, I_t) to be used later of inner trajectories
	MC_k<<<1,512,3*512*sizeof(float)>>>(Stock_tracker,x_0, tau, vol, dt, 
				  P1, P2, K, res1,I_tTracker , time_tracker,
				  B, Ntraj, Ntraj_inner, M, CMRG);
	

	
	//Simulation of inner trajectories, each one starting for a set of (Time, Stock, I_t) 
	// Generated by the outer trajectories kernel. 
	
	MC_inner_trajectory_k<<<512*100,1024,3*1024*sizeof(float)>>>(Stock_tracker,tau,
							vol, dt, P1, P2,
							K, inner_price_resultGPU, I_tTracker, time_tracker,
							B, Ntraj_inner, M,
							CMRG);
		

	//Wait for all Kernels to finish filling their results in inner_price_resultGPU. 
	cudaDeviceSynchronize();

	cudaEventRecord(stop,0);			// GPU timer instructions
	cudaEventSynchronize(stop);			// GPU timer instructions
	cudaEventElapsedTime(&Tim,			// GPU timer instructions
			 start, stop);				// GPU timer instructions
	cudaEventDestroy(start);			// GPU timer instructions
	cudaEventDestroy(stop);				// GPU timer instructions
	printf("Succes. Time elapsed = %f ms\n", Tim);	


	//Transfer the data (Time, Stock, I_t) into CPU 
	cudaMemcpy(S_t, Stock_tracker, Ntraj*M * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(t, time_tracker, Ntraj*M * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(I_t, I_tTracker, Ntraj*M * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&sum, res1, sizeof(float), cudaMemcpyDeviceToHost);
	

	// Transfer Nested price results to a CPU Array "inner_price_resultCPU"
	cudaMemcpy(inner_price_resultCPU, inner_price_resultGPU,Ntraj * M * sizeof(float), cudaMemcpyDeviceToHost);

	printf("Writing generated data to a .csv file.\n");
	//Writing the data generated into a ".csv" file
	float a,c,d ;
	int b ;
	FILE *outfile =fopen("Nested_MC_Bullet_Option.csv","w");
	fprintf(outfile,"Time,Stock,I_t,Price\n");
	fprintf(outfile,"0,100,0,%f\n",sum);
	for (int k = 0;k < Ntraj*M; ++k){
		a = S_t[k];
		b = I_t[k];
		c = t[k];
		d = inner_price_resultCPU[k];

		fprintf(outfile,"%f,%f,%i,%f\n",c,a,b,d);
	}
	fclose(outfile);
	
	printf("Freeing Memroy.\n");
	//Freeing all memeory used
	cudaFree(res1);
	cudaFree(time_tracker);
	cudaFree(Stock_tracker);
	cudaFree(I_tTracker);
	cudaFree(inner_price_resultGPU) ; 
	free(inner_price_resultCPU);
	FreeCMRG();
	
	return 0;
}


