function X = makef(w,Fq,Amp,Wid,aper)
%
% afit.makef(w,Fq,Amp,Wid)
%
% e.g.
% w = 1:100
% S = afit.makef(w,[10 20 50 70],[10 8 5 3],[1 2 5 3]);
% figure;plot(w,S)
%
% AS
doaperiod=0;

if isstruct(Fq)
    Amp = Fq.Amp;
    Wid = Fq.Wid;
    
    if isfield(Fq,'aper')
        aper = Fq.aper;
        doaperiod = 1;
    end
    
    Fq  = Fq.Freq;
end

if nargin == 5 && ~isempty(aper)
    doaperiod = 1;
end

if length(Fq) > 1
    for i = 1:length(Fq)
        if doaperiod && i == 1
            X0 =      makef(w,Fq(i),Amp(i),Wid(i),aper);
        else
            try
                X0 = X0 + makef(w,Fq(i),Amp(i),Wid(i));
            catch
                X0 =      makef(w,Fq(i),Amp(i),Wid(i));
            end
        end
        %X0(i,:) = afit.makef(w,Fq(i),Amp(i),Wid(i));

    end
    %X0 = max(X0); 
    X  = X0;
    return;
end


try Wid ; catch Wid = 2; end
try Amp ; catch Amp = 2; end

% offset negative values in w
mw  = min(w);
X   = 0*w;
f   = findthenearest(Fq,w);
f   = f(1);

w   = w - mw;
X   = X + Amp * exp( -(w-f).^2 / (2*(2*Wid)^2) );
w   = w + mw;

if sum(X) == 0
    % exception with 1 node models where sum==0
    X   = X + Amp * exp( -(w-Fq).^2 / (2*(2*Wid)^2) );
end

if doaperiod &&  length(Fq) == 1
    X = X + ( aper(1) * (w.^0) + w.^-aper(2) );
end