# aoptim

```
% A gradient/curvature descent optimisation routine, designed primarily 
% for nonlinear model fitting / system identification & parameter estimation. 
% 
% The objective function minimises the free energy (~ ELBO) or the SSE.
%
% Fit multivariate nonlinear models of the forms:
% 1)  Y0 = f(x) + e   (e.g. generative models) ..or
% 2)  e  = f(x)       (e.g. f() is the objective function)
%
% To optimise models using the FE, you need your code set up as per (1).
%
% Y0 = empirical data (to fit)
% f  = model (function)
% x  = model parameters/inputs to be optimised 
%      (treated as Gaussians with variance V)
%
% Given the model function (f), parameters (x) and data (y), AO computes the
% objective function, F:
%
%   F(x) = log evidence(e,y) - divergence(x) 
%
% For a multivariate function f(x) where x = [p1 .. pn]' the ascent scheme
% is:
%
%   x[p,t+1] = x[p,t] +         a[p]          *-dFdx[p]
%
% Note the use of different step sizes, a[p], for each parameter.
% Optionally, a second GD can be computed to find the best step size a[p]:
%
%   x[p,t+1] = x[p,t] + ( a[p] + b*-dFda[p] ) *-dFdx[p] ... where b = 1e-4
%
% For the new step sizes, the Armijo-Goldstein condition is enforced. The
% secondary optimisation (of a) is only invoked is the full gradient
% prediction initially inproved F, since it is computationally intensive.
%
% dFdx[p] are the partial derivatives of F, w.r.t parameters, p. (Note F = 
% the objective function and not 'f' - your function). See jaco.m for options, 
% although by default these are computed using a finite difference 
% approximation of the curvature, which retains the sign of the gradient:
%
% f0 = F(x[p]+h) 
% fx = F(x[p]  )
% f1 = F(x[p]-h) 
%
% j(p,:) =        (f0 - f1) / 2h  
%          ----------------------------
%            (f0 - 2 * fx + f1) / h^2  
%
% nb. - depending on the specified step method, a[] is computed from j & V
% using variations on:      (where V is the variance of each param)
% 
%  J      = -j ;
%  dFdpp  = -(J'*J);
%  a      = (V)./(1-dFdpp);   
% 
% If step_method = 3, dFdpp is a small number and this reduces to a simple
% scale on the stepsize - i.e. a = V./scale. However, if step_method = 1, J
% is transposed, such that dFdpp is np*np. This leads to much bigger
% parameter steps & can be useful when your start positions are a long way
% from the solution. In either case, note that the variance term V[p] on the
% parameter distribution is a scaled version of the step size. 
%
% For each iteration of the ascent:
% 
%   dx[p]  = x[p] + a[p]*-dFdx
%
% There are 4 possible outcomes for dx. In order:
%   
%   (1.) F(dx[1:np]) < F(x[1:np]) - i.e. all dx improve F ... keep going.
%   (2.) If (1.) fails, evaluate each dx[ip] and form subset, R, to update:
%        F(dx[R]) < F(x) ... keep going.
%   (3.) If 2 fails, do a very selective update: 
%        - Compute prob of each dx[ip] coming from Gauss defined by x[ip] & V[ip].
%        - Sort params by Prob(ip)*Improvment(ip)
%        - Iteratively accept each individual param update in order until
%          no more improvement.
%   (4.) If 1:3 fail, then reject dx, expand V and recompute derivatives. 
%        Go to (1).
%
% Usage: to minimise a model fitting problem of the form:
%==========================================================================
%   y    = f(p)                              ... f = function, p = params
%   e    = (data - y)                        ... error = data - f(p)
%   F(p) = log evidence(e,y) - divergence(p) ... objective function F
%
% the long usage is:
%   [X,F,Cp,Pp,Hist] = AO(f,x0,V,data,maxit,inner_loop,Q,crit,min_df,ordr,...
%                                writelog,obj,ba,im,step_meth)
%
% minimum usage (using defaults):
%   [X,F] = AO(fun,x0,V,[y])
%
%
% INPUTS:
%-------------------------------------------------------------------------
% To call this function using an options structure (recommended), do this:
%-------------------------------------------------------------------------
% opts = AO('options');   % get the options struct
% opts.fun = @myfun       % fill in what you want...
% opts.x0  = [0 1];       % start param value
% opts.V   = [1 1]/8;     % variances for each param
% opts.y   = [...];       % data to fit - .e.g y = f(p) + e
% opts.step_meth = 1;     % 
% [X,F] = AO(opts);       % call the optimser, passing the struct
%
% OUTPUTS:
%-------------------------------------------------------------------------
% X   = posterior parameters
% F   = fit value (depending on objective function specified)
% CP  = parameter covariance
% Pp  = posterior probabilites
% H   = history
%
% Notes: 
% (1) in my testing minimising SSE seems to produce the best overall fits but
% free energy (log evidence - divergence) converges much faster. 
% (2) the free energy equation is
%       L(1) = spm_logdet(iS)*nq/2  - real(e'*iS*e)/2 - ny*log(8*atan(1))/2;  ...
%       L(2) = spm_logdet(ipC*Cp)/2 - p'*ipC*p/2;
%       F    = -sum(L);
%
% *NOTE THAT, FOR FREE ENERGY OBJECTIVE, THE OUTPUT F-VALUE IS SIGN FLIPPED!
% *If the optimiser isn't working well, try making V smaller!

```

Here's a video of the the optimiser solving a system of nonlinear differential equations that describe a mean-field neural mass model - fitting it's spectral output to some real data:
```
![screenshot](OptimisationGIF.gif)
