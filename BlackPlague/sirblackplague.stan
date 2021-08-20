//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// The input data is a vector 'y' of length 'N'.


functions{
  real []SIR( real t, real[] X, real[] theta,
            real[] x_r, int[] x_i){
    real alpha;
    real beta;
    real dSIR_dt[3]; 
    real dSdt;
    real dIdt;
    real dRdt;
    
    alpha=theta[1];
    beta=theta[2];
    
    //X1= St, X2=It, X3=Rt
    
    dSdt=  -beta* X[1]*X[2];
    dIdt=   beta* X[1]*X[2]-alpha*X[2];
    dRdt=   alpha*X[2];
    
    dSIR_dt[1]=dSdt;
    dSIR_dt[2]=dIdt;
    dSIR_dt[3]=dRdt;
    
    return dSIR_dt;
              

  }
  
}

data {
  
  real t0;                     // initial time
  int<lower=0> N_t;            // number of measurement times
  real<lower=t0> ts[N_t];     // measurement times
  int yt[N_t];               //Removed R(t)=yt
  int xt[2];               //  Infected I(t)=xt
  int N;                   //total population
  real <lower=0> Inv_T;
}



transformed data {
  real x_r[0];                 // no real data for ODE system
  int x_i[0];                  // no integer data for ODE system
}

parameters {
  real<lower=0, upper=1> alpha;
  real<lower=0, upper=1> beta;
  real<lower=1, upper=13> I0;
}

transformed parameters{
  real E;
  {
    real Xhatt[N_t, 3];          // Output from the ODE solver
    real theta[2];            // ODE parameters
    real x0[3];              // Initial condition
    real logyhat[N_t];      // log of Binomial pdf for y
    real logxhat[2];      // log of Binomial pdf for x
    real pt;
    int s;
    E=0;
  
    theta[1]=alpha;
    theta[2]=beta;
    
    x0[1]=N-I0; //S0
    x0[2]=I0; //0
    x0[3]=0; //R0
   
    Xhatt = integrate_ode_rk45(SIR, x0, t0, ts, theta, x_r, x_i);
    
    for (t in 1: N_t) {
        if(Xhatt[t,3]> 261 )
          Xhatt[t,3]=261;
        
        pt=Xhatt[t,3]/N ;
        logyhat[t]= binomial_lpmf( yt[t] | N, pt  ); 
    }
    
    s=1;
    for(t in N_t-1: N_t){ 
        if(Xhatt[t,2]<0 )
            Xhatt[t,2]=0;
          
        pt=Xhatt[t,2]/N ;
        logxhat[s]= binomial_lpmf( xt[s] | N, pt  ); 
        s=s+1;
    }
    
    
    E=E-gamma_lpdf( alpha | 1,1  ); //prior for alpha
    E=E-gamma_lpdf( beta | 1,1 ); //prior for beta
    E=E-normal_lpdf(I0 | 5, 2.21 ); //prior for I0
    E=E-sum(logyhat)-sum(logxhat); //+1e-200 ; //log L
     
  }
  
  
}

model {
    
    target += -Inv_T * E;
}







