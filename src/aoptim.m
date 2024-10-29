function [x0,vx,e,iS] = aoptim(fun,x0,V,y,Nit)
% approximate Variational Laplace 
%
% [x1,e,iS] = aoptim(fun,x0,V,y,[num_it])
%
%
% AS2024

if nargin < 5 || isempty(Nit)
    Nit = 256;
end

figure('position',[1032         113        1305        1189]);

UseParallel = 0;

if UseParallel; 
    Jfun = @jaco_mimo_par;
else
    Jfun = @jaco_mimo;
end

% initial hyperparameter components
%--------------------------------------------------------------------------
[parts,moments]=iterate_gauss(y,2);
%rm = find(var(parts')<1e-3);
%parts(rm,:) = [];
%num_precision_comps = 16;
%parts = gaubasis(length(y),num_precision_comps);
for i = 1:size(parts,1)
    Q{i} = diag(parts(i,:)); 
end

ip  = find(sum(V,2));

hQ = parts'\y;
eh = hQ;
p0 = x0;

% initial position
f   = @(p,hyp,QQ,J) obj(fun,p,y,hyp,QQ,p0,V,J,eh);

% Compute derivatives for iteration 1
%--------------------------------------------------------------------------
[J] = Jfun(fun,x0,V,0,1);
if size(J{1},1) == 1; J   = cat(1,J{:})';
else;                 J   = cat(2,J{:});
end

for ij = 1:size(J,2)
    J(:,ij) = denan( J(:,ij) ./ norm(J(:,ij)) );
end
    
% recompute initial error with precomputed J
e   = f(x0,hQ,Q,J); ae = e;
r   = spm_vec(y) - spm_vec(fun(x0));

% re-estimate initial hyperparameter now we can obtain initial residual
hQ = parts'\abs(r);

% re-estimate initial error with J and initial h
e   = f(x0,hQ,Q,J); ae = e;

% x-vector for plotting and initial step-size scalar St
w   = 1:length(y);
St  = 1/8;

% initial hyperparameter 
iS = tQ(hQ,Q);

fprintf('iter %d; e = %d\n',0,e);

% BEGIN ITERATION LOOP
%--------------------------------------------------------------------------
for k = 1:Nit

    % gradient - parameters
    %-------------------------------------------------
    if k > 1
        [J] = Jfun(fun,x0,V,0,1);
        if size(J{1},1) == 1; J   = cat(1,J{:})';
        else;                 J   = cat(2,J{:});
        end

        for ij = 1:size(J,2)
            J(:,ij) = denan( J(:,ij) ./ norm(J(:,ij)) );
        end
    end

    % Residual, MAP solution & ~variance
    r  = spm_vec(y) - spm_vec(fun(x0));
    dx = x0 + V.*atcm.fun.aregress(iS*J,r,'MAP');

    %[u,v] = lu(J');
    %dx    = x0 + (u*u')\(u*v*r);

    vx = diag(inv((J'*iS*J)'));

    subplot(3,2,4);errplot(dx(ip),(vx(ip)));
    title('Parameters: Mean & Variance');drawnow;

    % compute objective function given dx
    m      = fun(dx);
    e      = obj(fun,dx,y,hQ,Q,p0,V,J,eh);

    if all(m==0)
        % revert; remove regularisation
        dx = x0 + St * spm_dx(dFdpp,dFdp);
        m  = fun(dx,hQ,Q);
    end

    % gradient - hyperparams
    %-------------------------------------------------
    h  = @(h) hobj(fun,dx,y,h,Q,p0,V,J,eh);
    %Jh = spm_diff(h,hQ,1); Jh = Jh./norm(Jh);

    % update hypers - a linear model
    r    = spm_vec(y) - spm_vec(fun(dx));
    %step = tdQ(hQ,Q)'\(r);
    
    [Mu,Cov,b,bv] = atcm.fun.agaussreg(tdQ(hQ,Q)',(r));
    step = b*r;
    dh   = hQ(:) - St * step(:);% hQ(:) - St * step(:);
    
    % if new estimate not better, iterate reducing step
    if h(dh) < h(hQ)
        erh = h(hQ) - h(dh);
        hQ = dh;
    else
        re = 1/8;
        while h(hQ) < h(dh)
            re  = re / 2;
            dh = hQ(:) - re*step(:);
        end
        erh = h(hQ) - h(dh);
        hQ = dh;
    end

    % full evaluation of objective; with dx and dh
    de = obj(fun,dx,y,hQ,Q,p0,V,J,eh); 
    
    % update
    if de > e
       fprintf('iter %d; e = %d, e(h) = %d\n',k,de,erh);
       St = St / 2;
    else
        fprintf('iter %d; e = %d, e(h) = %d\n',k,de,erh);
        x0  = dx;
        e   = de;
    end
    
    % explicitly get h-matrix for plot and to weight J on next step
    [~,~,~,~,iS] = atcm.fun.agaussreg(tdQ(hQ,Q)',abs(r));
    %iS = Mu'*diag(b)*Mu;
    %iS = tQ(hQ,Q);

    % update Q
    %for i = 1:length(Q)
    %    Q{i} = b(i) * diag(Mu(i,:));
    %end

    % and plot;
    ae = [ae; de];
    mx = fun(x0);
    subplot(3,2,1); plot(0:k,ae); title('Objective Function / Iterations');
    
    % upper and lowers on model prediction
    subplot(3,2,2); cla;
    subplot(3,2,2); plot(w,y,':',w,mx); title('Data & Model');

    subplot(3,2,3); imagesc(iS); title('Hyperparameters on Precision of Gaussians');
    subplot(3,2,5); errplot(b'*Mu,bv'*Cov);title('Mean & Variance: Residual');hold off;
    %subplot(3,2,6); plot(w,tdQ(hQ,Q)); title('Estimated Components');
    subplot(3,2,6); plot(w,b.*Mu); title('Estimated Components');
    drawnow;
    

end


end

function V = tdQ(h,Q)

for i = 1:length(h)
    V(i,:) = ( (h(i))) * diag(Q{i});
end

end

function iS = tQ(h,Q)

iS     = 0;
for i  = 1:length(h)
    iS = iS + ( (( (h(i))) * Q{i}) );
end

end

function y = hypupd(V)

y = VtoGauss(V);
y = y./norm(full(y));

end

function e = obj(fun,x0,y,h,Q,p0,V,J,eh,LB,UB)

%  obj(fun,p,y,hyp,QQ,p0,V,J,eh);

%if isvector(iS); iS = hypupd(iS); end

%iS     = 0;
%for i  = 1:length(h)
%    iS = iS + VtoGauss( diag((exp(-32) + exp(h(i))) * Q{i}) );
%end

iS = tQ(h,Q);

m = fun(x0);
r = (y(:) - m(:)).^2;

% accuracy model
e =  r(:)'*iS*r(:);





%e = KLDiv((iS*y)',(iS*m)');

% % accuracy params
% pe = x0 - p0;
% Cp  = full( spm_inv( (J'*iS*J)  ) + diag(full(V)) );
% 
% e_p = full(pe(:)'*pinv(Cp)*pe(:));
% 
% % accuracy hypers
% he = h - eh;
% 
% for i = 1:length(Q)
%     H(i,:) = spm_vec(diag(Q{i}));
% end
% 
% iH = pinv(full(H*H'));
% e_h = he(:)'*iH*he(:);

% then full objective is
%e = e + e_p + e_h;

end

function e = hobj(fun,x0,y,h,Q,p0,V,J,eh,LB,UB) % iS


m = fun(x0);

iS = tQ(h,Q);

r = (y(:) - m(:)).^2;


e =  r(:)'*iS*r(:);

% % accuracy params
% pe = x0 - p0;
% Cp  = full( spm_inv( (J'*iS*J)  ) + diag(full(V)) );
% 
% e_p = full(pe(:)'*pinv(Cp)*pe(:));
% 
% % accuracy hypers
% he = h - eh;
% 
% for i = 1:length(Q)
%     H(i,:) = spm_vec(diag(Q{i}));
% end
% 
% iH = pinv(full(H*H'));
% e_h = he(:)'*iH*he(:);

% then full objective is
%e = e + e_p + e_h;



%e = Dg(:)'*iS*Dg(:);
%eb  = trace(dgY*iS*dgY');

%e   = (e - eb)./eb;


end
