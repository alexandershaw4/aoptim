function [EP,F,CP] = AO_DCM(P,DCM)
global DD

DD   = DCM;
DD.P = P;
Px   = spm_vec(P);
pC   = DCM.M.pC;
pC   = diag(spm_vec(pC));

fprintf('Performing AO optimisation\n');


% reduction
V = spm_svd(pC);
c = V'*diag(pC);
p = V'*Px;
DD.V = V;

%Q = DCM.xY.Q;
%Q = HighResMeanFilt(Q,1,4);
%Q = full(Q)./sum(Q(:));

[X,F,CP]  = AO(@fakeDM,p',c,DCM.xY.y,128,12*4,[],1e-3);
EP        = spm_unvec(V*X,DCM.M.pE);

end

function [y] = fakeDM(Px,varargin)
global DD

V    = DD.V;
PP   = spm_unvec(V*Px',DD.P); 
IS   = spm_funcheck(DD.M.IS);       % Integrator
y    = IS(PP,DD.M,DD.xU);           % Prediction
y    = spm_vec(y);                  
y    = real(y);

end
