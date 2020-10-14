function GAN = ao_generator(M,goal)
% the generator component for turning a trained ff neural network
% (using ao_nn) into a sort of gan
%
% GAN = ao_generator(M,goal)
%
% where M is the trained neural network returned by ao_nn and goal is the
% desired output of the discriminator function (one of the already trained 
% outputs of M).
%
% The generative network is a mirror architecture of the discriminator, so
% if the trained network has the shape:
%       f(M.m,x) = x*m{1}*diag(1./(1+exp(-m{2})))*m{3});
% then the generative network has the shape
%       g(N.m,q) = q*m{3}'*diag(1./(1+exp(-m{2}')))*m{1}');
% the complete gan function is then
%       gan(p,q) = f(M.m, g(spm_unvec(p,N.m),q))
% the parameters (weights, p) of gan are optimised for [goal] (where goal
% is one of the trained outputs of the passed nn) while q is iteratively
% sampled from between [0,1] with re-optimisation on each sample.
%
% AS2020

% the generator model is the reversed ffnn
n = length(M.modelspace);
for i = 1:n
    m{i} = randn(size(M.modelspace{end-(i-1)}))';
end

p = spm_vec(m);
c = ~~p/8;

% the generator network
g  = @(m,x)(x*m{1}*diag(1./(1+exp(-m{2})))*m{3});
gg = @(p,x) g(spm_unvec(p,m),x);

% connect the generator and discriminator together!
gan = @(p,x) M.fun_nr(M.modelspace, gg(p,x) );

rng default

for i = 1:10
    
    Q  = rand(1,2);
    fg = @(p) gan(p,Q);
    
    % what we want the discriminator to find:
    %goal = [0 1];  % a patient!
    
    % free energy optimisation settings (required AO.m)
    op = AO('options');
    op.fun       = fg;
    op.x0        = p(:);
    op.V         = c(:);
    op.y         = {goal};
    op.maxit     = 30;
    op.criterion = -500;
    op.step_method  = 1;
    op.BTLineSearch = 0;
    op.hyperparams  = 1;
    op.inner_loop   = 8;
    
    [X,F,CP,Pp]  = AO(op);
    
    p = X;
end

GAN.model = m;
GAN.p     = X;
GAN.g     = g;
GAN.gg    = gg;
GAN.gan   = gan;
GAN.fg    = fg;
GAN.goal  = goal;
