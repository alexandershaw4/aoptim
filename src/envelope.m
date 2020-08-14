function [upperEnv,lowerEnv] = envelope(x, n, method)
%ENVELOPE Envelope detector.
%   [YUPPER,YLOWER] = ENVELOPE(X) returns the upper and lower envelopes of
%   the input sequence, X, using the magnitude of its analytic signal.
%   
%   The function initially removes the mean of X and restores it after
%   computing the envelopes.  If X is a matrix, ENVELOPE operates
%   independently over each column of X.
%
%   [YUPPER,YLOWER] = ENVELOPE(X,N) uses an N-tap Hilbert filter to compute
%   the upper envelope of X.
%
%   [YUPPER,YLOWER] = ENVELOPE(X,N,ENVTYPE) specifies the type of envelope
%   to return. The default is 'analytic':
%      'analytic' - returns the analytic envelope via an N-tap FIR filter
%      'rms'      - returns the RMS envelope of X over a sliding window
%                   of N samples.  
%      'peak'     - returns the peak envelope of the signal using a spline
%                   over local maxima separated by at least N points.
%
%   ENVELOPE(...) without output arguments plots the signal and the upper
%   and lower envelope.
%
%   % Example 1:
%   %   Plot the analytic envelope of a decaying sinusoid.
%   x = 1 + cos(2*pi*(0:999)'/20).*exp(-0.004*(0:999)');
%   envelope(x);
%
%   % Example 2:
%   %   Plot the analytic envelope of a decaying sinusoid using a filter
%   %   with 50 taps.
%   x = 1 + cos(2*pi*(0:999)'/20).*exp(-0.004*(0:999)');
%   envelope(x,50);
%
%   % Example 3:
%   %   Compute the moving RMS envelope of an audio recording of a train
%   %   whistle over every 150 samples.
%   load('train');
%   envelope(y,150,'rms');
%   
%   % Example 4:
%   %   Plot the upper and lower peak envelopes of a speech signal
%   %   smoothed over 30 sample intervals.
%   load('mtlb');
%   envelope(mtlb,30,'peak');
%   
%   See also: HILBERT RMS MAX MIN.

%   Copyright 2015-2018 The MathWorks, Inc.

%   References:
%     [1] Alan V. Oppenheim and Ronald W. Schafer, Discrete-Time
%     Signal Processing, 2nd ed., Prentice-Hall, Upper Saddle River, 
%     New Jersey, 1998.

narginchk(1,3);
nargoutchk(0,2);

% if nargin > 2
%     method = convertStringsToChars(method);
% end

validateattributes(x,{'single','double'},{'2d','real','finite'});

needsTranspose = isrow(x);
if needsTranspose
  x = x(:);
end

if nargin>1
  validateattributes(n,{'numeric'},{'integer','scalar','positive'}, ...
    'envelope','N',2);
end

% allow for 'peaks' (common typo) although it should be 'peak' envelope
if nargin>2
  method = validatestring(method,{'analytic','rms','peaks'});
else
  method = 'analytic';
end

if strcmpi(method,'peaks')
  % no need to remove DC bias from peak finding algorithm
  [yupper, ylower] = envPeak(x,n);
else
  % remove DC offset
  xmean = mean(x);
  xcentered = bsxfun(@minus,x,xmean);
  
  % compute envelope amplitude
  if nargin==1
    xampl = abs(hilbert(xcentered));
  elseif strcmpi(method,'analytic')
    xampl = envFIR(xcentered,n);
  elseif strcmpi(method,'rms')
    xampl = envRMS(xcentered,n);
  end
  
  % restore offset
  yupper = bsxfun(@plus,xmean,xampl);
  ylower = bsxfun(@minus,xmean,xampl);
end

if nargout==0
  plotEnvelope(x,yupper,ylower,method);
elseif needsTranspose
  upperEnv = yupper';
  lowerEnv = ylower';
else
  upperEnv = yupper;
  lowerEnv = ylower;
end

function plotEnvelope(x,yupper,ylower,method)
% plot each signal with a muted default color
% plot each envelope with the default color
colors = get(0,'DefaultAxesColorOrder');
for i=1:size(x,2)
  lineColor = colors(1+mod(i-1,size(colors,1)),:);
  if i==1
    hLine = plot(x(:,1));
    hAxes = ancestor(hLine,'axes');
    axesColor = get(hAxes,'Color');
    set(hLine,'Color',(axesColor+lineColor)/2);
  else
    line(1:size(x,1),x(:,i),'Color',(lineColor+axesColor)/2);
  end
  line(1:size(x,1),yupper(:,i),'Color',lineColor);
  line(1:size(x,1),ylower(:,i),'Color',lineColor);
end

% add legend only when one signal is present
if size(x,2)==1
  legend(getString(message('signal:envelope:Signal')), ...
         getString(message('signal:envelope:Envelope')));
end

% lookup and plot title string
catStrs = {'Analytic','RMS','Peak'};
catStr = catStrs{strcmp(method,{'analytic','rms','peaks'})};
titleStr = getString(message(['signal:envelope:' catStr 'Envelope']));
title(titleStr);


function y = envFIR(x,n)

% construct ideal hilbert filter truncated to desired length
fc = 1;
t = fc/2 * ((1-n)/2:(n-1)/2)';

hfilt = sinc(t) .* exp(1i*pi*t);

% multiply ideal filter with tapered window
beta = 8;
firFilter = hfilt .* kaiser(n,beta);
firFilter = firFilter / sum(real(firFilter));

% apply filter and take the magnitude
y = zeros(size(x),'like',x);
for chan=1:size(x,2)
  y(:,chan) = abs(conv(x(:,chan),firFilter,'same'));
end

% compute RMS 
function y = envRMS(x,n)
y = movrms(x,n,'same');

function [yupper,ylower] = envPeak(x,n)

% pre-allocate space for results
nx = size(x,1);
yupper = zeros(size(x),'like',x);
ylower = zeros(size(x),'like',x);

% handle default case where not enough input is given
if nx < 2
  yupper = x;
  ylower = x;
  return
end

% compute upper envelope
for chan=1:size(x,2)
  if nx > n+1
    % find local maxima separated by at least N samples
    [~,iPk] = findpeaks(double(x(:,chan)),'MinPeakDistance',n);
    %[~,iPk] = findpeaks(double(x(:,chan)),'NPeaks',n);
  else
    iPk = [];
  end
  
  if numel(iPk)<2
    % include the first and last points
    iLocs = [1; iPk; nx];
  else
    iLocs = iPk;
  end

  % smoothly connect the maxima via a spline.
  yupper(:,chan) = interp1(iLocs,x(iLocs,chan),(1:nx)','spline');
end

% compute lower envelope
for chan=1:size(x,2)
  if nx > n+1
    % find local minima separated by at least N samples
    [~,iPk] = findpeaks(double(-x(:,chan)),'MinPeakDistance',n);
    %[~,iPk] = findpeaks(double(-x(:,chan)),'NPeaks',n);
  else
    iPk = [];
  end
  
  if numel(iPk)<2
    % include the first and last points
    iLocs = [1; iPk; nx];
  else
    iLocs = iPk;
  end
  
  % smoothly connect the minima via a spline.
  ylower(:,chan) = interp1(iLocs,x(iLocs,chan),(1:nx)','spline');
end

