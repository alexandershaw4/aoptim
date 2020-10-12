function [X,F,CP,Pp] = ao_cnn(y,x,l,niter,ncomp)
% y = group identifier (0,1), size(n,1)
% x = predictor variances, size(n,m)
% l = convolutional layers sizes, e.g. l = {[32 4],[4 1]}
%
global aocnn

rng default;

[u,s,v] = spm_svd(x');

%if nargin ==5 && ~isempty(ncomp)
    x = x*u(:,1:ncomp);
% else
%     i = findthenearest(full(cumsum(diag(s))./sum(diag(s))),0.9);
%     x = x*u(:,1:i);
% end

[nob,np] = size(x); % num obs and vars


for i = 1:length(l)
    m{i} = ones(l{i}(1),l{i}(2)) / prod(l{i});
    %m{i} = rand(l{i}(1),l{i}(2)) / (prod(l{i}).^2);
    %m{i} = ones(l{i}(1),l{i}(2)) / prod(l{i});
end

[u,s,v] = spm_svd(real(x));

m{1} = full(v(:,1:(l{1}(2))));
x    = u;

aocnn.m  = m;
aocnn.n  = nob;
aocnn.np = np;
aocnn.x  = x;
aocnn.y  = y;

for i = 1:size(y,2)
    y(:,i) = 1 + ( y(:,i)/max(y(:,i)) );
end

%     the conv mats                the activations
p = [ ones(length(spm_vec(m)),1) ; ones(length(l),1) ];
c = p;

% gen(p);

op = AO('options');
op.fun = @gen;
op.x0 = p(:);
op.V = c(:);
op.y = {y};
op.maxit = niter;
op.criterion = -inf;
op.step_method=1;
op.BTLineSearch=0;
op.hyperparams=1;
[X,F,CP,Pp] = AO(op);

%[X,F,CP,Pp,History]  = AO(@gen,p(:),c,{y},niter,12*4,[],-inf,1e-12,0,2,0,'fe',0,1);


end

function pred = gen(p)
global aocnn

p  = p(:);
lm = length(spm_vec(aocnn.m));
m  = spm_unvec(p(1:lm),aocnn.m);
a  = p(lm+1:end);

for i = 1:length(m)
    m{i} = m{i} * act(a(i)) ;
end

% Run it
for s = 1:size(aocnn.x,1)
    xx = aocnn.x(s,:);
    for i = 1:length(m)
        xx = xx*m{i};
    end
    pred(s,:) = xx;
end

pred = {(pred)};

end

function y = act(x)

%a = 1;
%c = 0;
%y = 1./(1 + exp(-a.*(x-c)));

y = 1./(1 + exp(-x));

end