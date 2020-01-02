% example_aoglm
%
% Example of fitting a GLM using AO.m curvature-descent optimsation.
% Comparison of returned model (�) to that using glmfit.m

dat = load([fileparts(fileparts(mfilename('fullpath'))) '/src/test_glm_data']); 

% [disclaimer: somewhat unconventionally I have called the variable of
% predictor data 'y' and the thing we're predicting 'x' ]

% run the AO glm:
[b,F,cp,fit] = ao_glm(dat.x,dat.y);

% run glmfit
B = glmfit(dat.y,dat.x,'normal');

% take a look at the betas:
[B' ; b]