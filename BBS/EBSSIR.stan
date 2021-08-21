// English Boarding Schol Stan
//A Stan program is organized into a sequence of named blocks, 
//the bodies of which consist of variable declarations, 
//followed in the case of some blocks with statements


functions{
  //Define the SIR function here
  real []SIR( real t, real[] X, real[] theta,
            real[] x_r, int[] x_i){
    // Declare variables. Notice that it ends with a  ;
    
    real beta;
    real gamma;
    real dSIR_dt[3]; //This is a real array of size 3
    real dSdt;
    real dIdt;
    real dRdt;
    int N;
    
    N= x_i[1];
    beta=theta[1]; //indexingstarts in 1
    gamma=theta[2];
    
    dSdt=  -beta/N* X[1]*X[2];
    dIdt=   beta/N* X[1]*X[2]-gamma*X[2];
    dRdt=   gamma*X[2];
    
    dSIR_dt[1]=dSdt; //assign to components of the array
    dSIR_dt[2]=dIdt;
    dSIR_dt[3]=dRdt;
    
    return dSIR_dt;
            
  }
  
}

data {
  
  real t0;                     // initial time
  int<lower=0> N_t;            // number of measurement times
  real<lower=t0> ts[N_t];     // arrray measurement times
  int yt[N_t];               // yt is of size N_t
  int N;                   //total population
}

transformed data {
  // Not too important. Just need to specify it bc of the definition 
  // of the SIR function.
  real x_r[0];                 // no real data for ODE system
  int x_i[1];                  // Pass N as integer data for ODE system
  x_i[1] = N;
}

parameters {
  real<lower=0> beta; // Support of parameters can be defined inside declaration of variable
  real<lower=0> gamma; 
  real<lower=0, upper=1> i0;// Initial condition: fraction of initially infected
}


//The solutions to the SIR equations for a given initial state 
// are defined as transformed parameters. This will allow them to be used in the model and inspected
// in the output. It also makes it clear that they are all functions of the initial population and parameters (as well as the solution times).

transformed parameters{
  real Shat[N_t, 3]; //Solution of ODE
  real theta[2];          // ODE parameters: beta= theta[1], gamma=theta[2]
  real x0[3];              // Initial condition
  
  theta[1]=beta;
  theta[2]=gamma;
    
  // Initial Condition
  x0[1]=N-i0*N  ; //S0
  x0[2]=i0*N; //I0
  x0[3]=0; //R0
   
   //Solution of Ode for specified params
  Shat = integrate_ode_rk45(SIR, x0, t0, ts, theta, x_r, x_i); 
    
  
}



model {
    //priors
    beta ~ lognormal(0,1);
    gamma ~ gamma(0.02, 0.004);
    i0 ~ beta(0.5, 0.5);
    
    //likelihood
    yt ~ poisson(  Shat[,2]  );
    
}


generated quantities {
  real Srep[N_t];
  int Irep[N_t];
  real Rrep[N_t];
  for (n in 1:N_t){
      Srep[n]= Shat[n,1];
      Irep[n] = poisson_rng(  Shat[n,2]  );
      Rrep[n]= Shat[n,3];
  }
  }









