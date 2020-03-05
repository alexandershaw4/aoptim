function [b,F,Cp,fit] = ao_glm_eig(x,y,nc)
% use AO curvature optimiser to fit a (logistic?) regression/GLM...
%
% usgae: [b,F,Cp,fit] = ao_glm_eig(group,observations)
% - when nc == 3,
% - fits the lm: d = W(1:nc,:)*y
%                X = c + d(1)*ß(1) + d(2)*ß(2) + d(nc)*ß(nc)
%
% - performs similarly to glmfit
% - no orthogonalisation version
% - conceptually similar to PLS regression
%
% the betas are returned in fit.mb, such that the prediction is:
%  pred = fit.mb{1}+fit.mb{2}*fit.mb{3}*observations'
%
% the reduced data vectors as returned in fit.W.
%
% Plot the output:
% scatter3(fit.W(:,1),fit.W(:,2),fit.W(:,3),110,group,'filled')
%
% AS


% model:  d = W(1:nc,:)*y
%         x = c + d(1)*b(1) + d(2)*b(2) + d(nc)*b(nc)
%----------------------------------------------------
nb = size(y,2);
n  = size(y,1);

W  = ones(nc,nb); % d = y*W'
b  = [0 ones(1,nc) W(:)'];
V  = [2 b(2:end)/nb*nc];

fun = @(b) (b(1)+b(2:nc+1)*reshape(b(nc+2:end),[nc nb])*y');
[b,F,Cp] = AO(fun,b,V,x,inf,[],[],1e-8);

% compute betas on the non-orthongal predictors
%----------------------------------------------------
b = b';
m = b(1) + b(2:nc+1)*reshape(b(nc+2:end),[nc nb])*(y');

% assess fit using adjusted r^2
%----------------------------------------------------
[fit.r,fit.p] = corr(x(:),m(:)); 
fit.r2        = (fit.r).^2;
fit.ar2       = 1 - (1 - fit.r2) * (n - 1)./(n - nb - 1);

% reshape betas -
fit.mb{1} = b(1);
fit.mb{2} = b(2:nc+1);
fit.mb{3} = reshape(b(nc+2:end),[nc nb]);

% the reduced vectors
fit.W = (fit.mb{3}*y')';


% % explicitly compute fitted variables to get correlations
% %----------------------------------------------------
% for i = 1:nb
%     v0(:,i) = b(i+1)*y(:,i);
% end
% 
% % fitted predictor correlations
% %----------------------------------------------------
% [fit.fitted_par_r,fit.fitted_par_p]=corr(x,v0);
% fit.fitted_params = v0;
