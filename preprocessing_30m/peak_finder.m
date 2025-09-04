function [Ypk,Xpk,Wpk,Ppk] = peak_finder(Yin,varargin)

% extract the parameters from the input argument list
[y,yIsRow,x,xIsRow,minH,minP,minW,maxW,minD,minT,maxN,sortDir,annotate,refW] ...
  = parse_inputs(Yin,varargin{:});

% indicate if we need to compute the extent of a peak
needWidth = minW>0 || maxW<inf || minP>0 || nargout>2 || strcmp(annotate,'extents');

% find indices of all finite and infinite peaks and the inflection points
[iFinite,iInfinite,iInflect] = getAllPeaks(y);

% keep only the indices of finite peaks that meet the required
% minimum height and threshold
iPk = removePeaksBelowMinPeakHeight(y,iFinite,minH,refW);
iPk = removePeaksBelowThreshold(y,iPk,minT);

if needWidth
  % obtain the indices of each peak (iPk), the prominence base (bPk), and
  % the x- and y- coordinates of the peak base (bxPk, byPk) and the width
  % (wxPk)
  [iPk,bPk,bxPk,byPk,wxPk] = findExtents(y,x,iPk,iFinite,iInfinite,iInflect,minP,minW,maxW,refW);
else
  % combine finite and infinite peaks into one list
  [iPk,bPk,bxPk,byPk,wxPk] = combinePeaks(iPk,iInfinite);
end

% find the indices of the largest peaks within the specified distance
idx = findPeaksSeparatedByMoreThanMinPeakDistance(y,x,iPk,minD);

% re-order and bound the number of peaks based upon the index vector
idx = orderPeaks(y,iPk,idx,sortDir);
idx = keepAtMostNpPeaks(idx,maxN);

% use the index vector to fetch the correct peaks.
iPk = iPk(idx);
if needWidth
  [bPk, bxPk, byPk, wxPk] = fetchPeakExtents(idx,bPk,bxPk,byPk,wxPk);
end

if nargout > 0
  % assign output variables
  if needWidth
    [Ypk,Xpk,Wpk,Ppk] = assignFullOutputs(y,x,iPk,wxPk,bPk,yIsRow,xIsRow);
  else
    [Ypk,Xpk] = assignOutputs(y,x,iPk,yIsRow,xIsRow);
  end    
end

%--------------------------------------------------------------------------
function [y,yIsRow,x,xIsRow,Ph,Pp,Wmin,Wmax,Pd,Th,NpOut,Str,Ann,Ref] = parse_inputs(Yin,varargin)

% Validate input signal
y = Yin(:);
M = length(y);

yIsRow = isrow(Yin);

% indicate if the user specified an Fs or X
hasX = ~isempty(varargin) && (isnumeric(varargin{1}) || ...
    length(varargin{1})>1);
  %isInMATLAB && isdatetime(varargin{1}) && length(varargin{1})>1);

if hasX
  startArg = 2;
  FsSupplied = isscalar(varargin{1});
  if FsSupplied
    % Fs
    Fs = varargin{1};
    x = (0:M-1).'/Fs;
    xIsRow = yIsRow;
  else
    % X
    Xin = varargin{1};
    
    xIsRow = isrow(Xin);
    x = Xin(:);
  end
else
  startArg = 1;
  % unspecified, use index vector
  x = (1:M).';
  xIsRow = yIsRow;
end

defaultMinPeakHeight = -inf;
defaultMinPeakProminence = 0;
defaultMinPeakWidth = 0;
defaultMaxPeakWidth = Inf;
defaultMinPeakDistance = 0;
defaultThreshold = 0;
defaultNPeaks = [];
defaultSortStr = 'none';
defaultAnnotate = 'peaks';
defaultWidthReference = 'halfprom';

p = inputParser;
addParameter(p,'MinPeakHeight',defaultMinPeakHeight);
addParameter(p,'MinPeakProminence',defaultMinPeakProminence);
addParameter(p,'MinPeakWidth',defaultMinPeakWidth);
addParameter(p,'MaxPeakWidth',defaultMaxPeakWidth);
addParameter(p,'MinPeakDistance',defaultMinPeakDistance);
addParameter(p,'Threshold',defaultThreshold);
addParameter(p,'NPeaks',defaultNPeaks);
addParameter(p,'SortStr',defaultSortStr);
addParameter(p,'Annotate',defaultAnnotate);
addParameter(p,'WidthReference',defaultWidthReference);
parse(p,varargin{startArg:end});
Ph = p.Results.MinPeakHeight;
Pp = p.Results.MinPeakProminence;
Wmin = p.Results.MinPeakWidth;
Wmax = p.Results.MaxPeakWidth;
Pd = p.Results.MinPeakDistance;
Th = p.Results.Threshold;
Np = p.Results.NPeaks;
Str = p.Results.SortStr;
Ann = p.Results.Annotate;
Ref = p.Results.WidthReference;

% limit the number of peaks to the number of input samples
if isempty(Np)
    NpOut = M;
else
    NpOut = Np;
end

% ignore peaks below zero when using halfheight width reference
if strcmp(Ref,'halfheight')
  Ph = max(Ph,0);
end

%--------------------------------------------------------------------------
function [iPk,iInf,iInflect] = getAllPeaks(y)
% fetch indices all infinite peaks
iInf = find(isinf(y) & y>0);

% temporarily remove all +Inf values
yTemp = y;
yTemp(iInf) = NaN;

% determine the peaks and inflection points of the signal
[iPk,iInflect] = findLocalMaxima(yTemp);


%--------------------------------------------------------------------------
function [iPk, iInflect] = findLocalMaxima(yTemp)
% bookend Y by NaN and make index vector
yTemp = [NaN; yTemp; NaN];
iTemp = (1:length(yTemp)).';

% keep only the first of any adjacent pairs of equal values (including NaN).
yFinite = ~isnan(yTemp);
iNeq = [1; 1 + find((yTemp(1:end-1) ~= yTemp(2:end)) & ...
                    (yFinite(1:end-1) | yFinite(2:end)))];
iTemp = iTemp(iNeq);

% take the sign of the first sample derivative
s = sign(diff(yTemp(iTemp)));

% find local maxima
iMax = 1 + find(diff(s)<0);

% find all transitions from rising to falling or to NaN
iAny = 1 + find(s(1:end-1)~=s(2:end));

% index into the original index vector without the NaN bookend.
iInflect = iTemp(iAny)-1;
iPk = iTemp(iMax)-1;



%--------------------------------------------------------------------------
function iPk = removePeaksBelowMinPeakHeight(Y,iPk,Ph,widthRef)
if ~isempty(iPk) 
  iPk = iPk(Y(iPk) > Ph);
end
    
%--------------------------------------------------------------------------
function iPk = removePeaksBelowThreshold(Y,iPk,Th)
base = max(Y(iPk-1),Y(iPk+1));
iPk = iPk(Y(iPk)-base >= Th);

%--------------------------------------------------------------------------
function [iPk,bPk,bxPk,byPk,wxPk] = findExtents(y,x,iPk,iFin,iInf,iInflect,minP,minW,maxW,refW)
% temporarily filter out +Inf from the input
yFinite = y;
yFinite(iInf) = NaN;

% get the base and left and right indices of each prominence base
[bPk,iLB,iRB] = getPeakBase(yFinite,iPk,iFin,iInflect);

% keep only those indices with at least the specified prominence
[iPk,bPk,iLB,iRB] = removePeaksBelowMinPeakProminence(yFinite,iPk,bPk,iLB,iRB,minP);

% get the x-coordinates of the half-height width borders of each peak
[wxPk,iLBh,iRBh] = getPeakWidth(yFinite,x,iPk,bPk,iLB,iRB,refW);

% merge finite and infinite peaks together into one list
[iPk,bPk,bxPk,byPk,wxPk] = combineFullPeaks(y,x,iPk,bPk,iLBh,iRBh,wxPk,iInf);

% keep only those in the range minW < w < maxW
[iPk,bPk,bxPk,byPk,wxPk] = removePeaksOutsideWidth(iPk,bPk,bxPk,byPk,wxPk,minW,maxW);


%--------------------------------------------------------------------------
function [peakBase,iLeftSaddle,iRightSaddle] = getPeakBase(yTemp,iPk,iFin,iInflect)
% determine the indices that border each finite peak
[iLeftBase, iLeftSaddle] = getLeftBase(yTemp,iPk,iFin,iInflect);
[iRightBase, iRightSaddle] = getLeftBase(yTemp,flipud(iPk),flipud(iFin),flipud(iInflect));
iRightBase = flipud(iRightBase);
iRightSaddle = flipud(iRightSaddle);
peakBase = max(yTemp(iLeftBase),yTemp(iRightBase));

%--------------------------------------------------------------------------
function [iBase, iSaddle] = getLeftBase(yTemp,iPeak,iFinite,iInflect)
IZERO = getIZERO;
IONE = ones('like',IZERO);

% pre-initialize output base and saddle indices
iBase = zeros(size(iPeak),'like',IZERO);
iSaddle = zeros(size(iPeak),'like',IZERO);

% table stores the most recently encountered peaks in order of height
peak = zeros(size(iFinite));
valley = zeros(size(iFinite));
iValley = zeros(size(iFinite),'like',IZERO);

n = IZERO;
i = IONE;
j = IONE;
k = IONE;

% pre-initialize v for code generation
v = nan('like',yTemp); 
iv = IONE;

while k<=length(iPeak)
  % walk through the inflections until you reach a peak
  while iInflect(i) ~= iFinite(j) 
    v(1) = yTemp(iInflect(i));
    iv = iInflect(i);
    if isnan(v)
      % border seen, start over.
      n = IZERO;
    else
      % ignore previously stored peaks with a valley larger than this one
      while n>0 && valley(n)>v
        n = n - 1;
      end
    end
    i = i + 1;
  end
  % get the peak
  p = yTemp(iInflect(i));
  
  % keep the smallest valley of all smaller peaks
  while n>0 && peak(n) < p
    if valley(n) < v
      v(1) = valley(n);
      iv = iValley(n);
    end
    n = n - 1;
  end

  % record "saddle" valleys in between equal-height peaks
  isv = iv;
  
  % keep seeking smaller valleys until you reach a larger peak
  while n>0 && peak(n) <= p
    if valley(n) < v
      v(1) = valley(n);
      iv = iValley(n);
    end
    n = n - 1;      
  end
  
  % record the new peak and save the index of the valley into the base
  % and saddle
  n = n + 1;
  peak(n) = p;
  valley(n) = v;
  iValley(n) = iv;

  if iInflect(i) == iPeak(k)
    iBase(k) = iv;
    iSaddle(k) = isv;
    k = k + 1;
  end
  
  i = i + 1;
  j = j + 1;
end

%--------------------------------------------------------------------------
function [iPk,pbPk,iLB,iRB] = removePeaksBelowMinPeakProminence(y,iPk,pbPk,iLB,iRB,minP)
% compute the prominence of each peak
Ppk = y(iPk)-pbPk;

% keep those that are above the specified prominence
idx = find(Ppk >= minP);
iPk = iPk(idx);
pbPk = pbPk(idx);
iLB = iLB(idx);
iRB = iRB(idx);

%--------------------------------------------------------------------------
function [wxPk,iLBh,iRBh] = getPeakWidth(y,x,iPk,pbPk,iLB,iRB,wRef)
if isempty(iPk)
  % no peaks.  define empty containers
  base = zeros(size(iPk),'like',y);
  IZERO = getIZERO;
  iLBh = zeros(size(iPk),'like',IZERO);
  iRBh = zeros(size(iPk),'like',IZERO);  
elseif strcmp(wRef,'halfheight')
  % set the baseline to zero
  base = zeros(size(iPk),'like',y);

  % border the width by no more than the lowest valley between this peak
  % and the next peak
  iLBh = [iLB(1); max(iLB(2:end),iRB(1:end-1))];
  iRBh = [min(iRB(1:end-1),iLB(2:end)); iRB(end)];
  iGuard = iLBh > iPk;
  iLBh(iGuard) = iLB(iGuard);
  iGuard = iRBh < iPk;
  iRBh(iGuard) = iRB(iGuard);
else
  % use the prominence base
  base = pbPk;
  
  % border the width by the saddle of the peak
  iLBh = iLB;
  iRBh = iRB;
end

% get the width boundaries of each peak
wxPk = getHalfMaxBounds(y, x, iPk, base, iLBh, iRBh);

%--------------------------------------------------------------------------
function bounds = getHalfMaxBounds(y, x, iPk, base, iLB, iRB)
n = length(iPk);
if isnumeric(x)
  bounds = zeros(n,2,'like',x);
else
  bounds = [x(1:n) x(1:n)];
end
% interpolate both the left and right bounds clamping at borders
for i=1:n
  
  % compute the desired reference level at half-height or half-prominence
  refHeight = (y(iPk(i))+base(i))/2;
  
  % compute the index of the left-intercept at half max
  iLeft = findLeftIntercept(y, iPk(i), iLB(i), refHeight);
  if iLeft < iLB(i)
    bounds(i,1) = x(iLB(i));
  else
    bounds(i,1) = linterp(x(iLeft),x(iLeft+1),y(iLeft),y(iLeft+1),y(iPk(i)),base(i));
  end
  
  % compute the index of the right-intercept
  iRight = findRightIntercept(y, iPk(i), iRB(i), refHeight);
  if iRight > iRB(i)
    bounds(i,2) = x(iRB(i));
  else
    bounds(i,2) = linterp(x(iRight), x(iRight-1), y(iRight), y(iRight-1), y(iPk(i)),base(i));
  end

end

%--------------------------------------------------------------------------
function idx = findLeftIntercept(y, idx, borderIdx, refHeight)
% decrement index until you pass under the reference height or pass the
% index of the left border, whichever comes first
while idx>=borderIdx && y(idx) > refHeight
  idx = idx - 1;
end

%--------------------------------------------------------------------------
function idx = findRightIntercept(y, idx, borderIdx, refHeight)
% increment index until you pass under the reference height or pass the
% index of the right border, whichever comes first
while idx<=borderIdx && y(idx) > refHeight
  idx = idx + 1;
end

%--------------------------------------------------------------------------
function xc = linterp(xa,xb,ya,yb,yc,bc)
% interpolate between points (xa,ya) and (xb,yb) to find (xc, 0.5*(yc-yc)).
xc = xa + (xb-xa) .* (0.5*(yc+bc)-ya) ./ (yb-ya);

% invoke L'Hospital's rule when -Inf is encountered. 
if isnumeric(xc) && isnan(xc)% || isdatetime(xc) && isnat(xc)
  % yc and yb are guaranteed to be finite. 
  if isinf(bc)
    % both ya and bc are -Inf.
    if isnumeric(xa)
      xc(1) = 0.5*(xa+xb);
    else
      xc(1) = xa+0.5*(xb-xa);
    end
  else
    % only ya is -Inf.
    xc(1) = xb;
  end
end

%--------------------------------------------------------------------------
function [iPk,bPk,bxPk,byPk,wxPk] = removePeaksOutsideWidth(iPk,bPk,bxPk,byPk,wxPk,minW,maxW)

if isempty(iPk) || minW==0 && maxW == inf
  return
end

% compute the width of each peak and extract the matching indices
w = diff(wxPk,1,2);
idx = find(minW <= w & w <= maxW);

% fetch the surviving peaks
iPk = iPk(idx);
bPk = bPk(idx);
bxPk = bxPk(idx,:);
byPk = byPk(idx,:);
wxPk = wxPk(idx,:);

%--------------------------------------------------------------------------
function [iPkOut,bPk,bxPk,byPk,wxPk] = combinePeaks(iPk,iInf)
iPkOut = union(iPk,iInf);
bPk = zeros(0,1);
bxPk = zeros(0,2);
byPk = zeros(0,2);
wxPk = zeros(0,2);

%--------------------------------------------------------------------------
function [iPkOut,bPkOut,bxPkOut,byPkOut,wxPkOut] = combineFullPeaks(y,x,iPk,bPk,iLBw,iRBw,wPk,iInf)
iPkOut = union(iPk, iInf);

% create map of new indices to old indices
[~, iFinite] = intersect(iPkOut,iPk);
[~, iInfinite] = intersect(iPkOut,iInf);

% prevent row concatenation when iPk and iInf both have less than one
% element
iPkOut = iPkOut(:);

% compute prominence base
bPkOut = zeros(size(iPkOut));
bPkOut(iFinite) = bPk;
bPkOut(iInfinite) = 0;

% compute indices of left and right infinite borders
iInfL = max(1,iInf-1);
iInfR = min(iInf+1,length(x));

% copy out x- values of the left and right prominence base
% set each base border of an infinite peaks halfway between itself and
% the next adjacent sample
if isnumeric(x)
  bxPkOut = zeros(size(iPkOut,1),2);
  bxPkOut(iFinite,1) = x(iLBw);
  bxPkOut(iFinite,2) = x(iRBw);
  bxPkOut(iInfinite,1) = 0.5*(x(iInf)+x(iInfL));
  bxPkOut(iInfinite,2) = 0.5*(x(iInf)+x(iInfR));
else
  bxPkOut = [x(1:size(iPkOut,1)) x(1:size(iPkOut,1))];
  bxPkOut(iFinite,1) = x(iLBw);
  bxPkOut(iFinite,2) = x(iRBw);
  bxPkOut(iInfinite,1) = x(iInf) + 0.5*(x(iInfL)-x(iInf));
  bxPkOut(iInfinite,2) = x(iInf) + 0.5*(x(iInfR)-x(iInf));
end

% copy out y- values of the left and right prominence base
byPkOut = zeros(size(iPkOut,1),2);
byPkOut(iFinite,1) = y(iLBw);
byPkOut(iFinite,2) = y(iRBw);
byPkOut(iInfinite,1) = y(iInfL);
byPkOut(iInfinite,2) = y(iInfR);

% copy out x- values of the width borders
% set each width borders of an infinite peaks halfway between itself and
% the next adjacent sample
if isnumeric(x)
  wxPkOut = zeros(size(iPkOut,1),2);
  wxPkOut(iFinite,:) = wPk;
  wxPkOut(iInfinite,1) = 0.5*(x(iInf)+x(iInfL));
  wxPkOut(iInfinite,2) = 0.5*(x(iInf)+x(iInfR));
else
  wxPkOut = [x(1:size(iPkOut,1)) x(1:size(iPkOut,1))];
  wxPkOut(iFinite,:) = wPk;
  wxPkOut(iInfinite,1) = x(iInf)+0.5*(x(iInfL)-x(iInf));
  wxPkOut(iInfinite,2) = x(iInf)+0.5*(x(iInfR)-x(iInf));
end

%--------------------------------------------------------------------------
function idx = findPeaksSeparatedByMoreThanMinPeakDistance(y,x,iPk,Pd)
% Start with the larger peaks to make sure we don't accidentally keep a
% small peak and remove a large peak in its neighborhood. 

if isempty(iPk) || Pd==0
  IONE = ones('like',getIZERO);
  idx = (IONE:length(iPk)).';
  return
end

% copy peak values and locations to a temporary place
pks = y(iPk);
locs = x(iPk);

% Order peaks from large to small
[~, sortIdx] = sort(pks,'descend');
locs_temp = locs(sortIdx);

idelete = ones(size(locs_temp))<0;
for i = 1:length(locs_temp)
  if ~idelete(i)
    % If the peak is not in the neighborhood of a larger peak, find
    % secondary peaks to eliminate.
    idelete = idelete | (locs_temp>=locs_temp(i)-Pd)&(locs_temp<=locs_temp(i)+Pd); 
    idelete(i) = 0; % Keep current peak
  end
end

% report back indices in consecutive order
idx = sort(sortIdx(~idelete));



%--------------------------------------------------------------------------
function idx = orderPeaks(Y,iPk,idx,Str)

if isempty(idx) || strcmp(Str,'none')
  return
end

[~,s]  = sort(Y(iPk(idx)),Str);

idx = idx(s);


%--------------------------------------------------------------------------
function idx = keepAtMostNpPeaks(idx,Np)

if length(idx)>Np
  idx = idx(1:Np);
end

%--------------------------------------------------------------------------
function [bPk,bxPk,byPk,wxPk] = fetchPeakExtents(idx,bPk,bxPk,byPk,wxPk)
bPk = bPk(idx);
bxPk = bxPk(idx,:);
byPk = byPk(idx,:);
wxPk = wxPk(idx,:);

%--------------------------------------------------------------------------
function [YpkOut,XpkOut] = assignOutputs(y,x,iPk,yIsRow,xIsRow)

% fetch the coordinates of the peak
Ypk = y(iPk);
Xpk = x(iPk);

% preserve orientation of Y
if yIsRow
  YpkOut = Ypk.';
else
  YpkOut = Ypk;
end

% preserve orientation of X
if xIsRow
  XpkOut = Xpk.';
else
  XpkOut = Xpk;
end

%--------------------------------------------------------------------------
function [YpkOut,XpkOut,WpkOut,PpkOut] = assignFullOutputs(y,x,iPk,wxPk,bPk,yIsRow,xIsRow)

% fetch the coordinates of the peak
Ypk = y(iPk);
Xpk = x(iPk);

% compute the width and prominence
Wpk = diff(wxPk,1,2);
Ppk = Ypk-bPk;

% preserve orientation of Y (and P)
if yIsRow
  YpkOut = Ypk.';
  PpkOut = Ppk.';
else
  YpkOut = Ypk;
  PpkOut = Ppk;  
end

% preserve orientation of X (and W)
if xIsRow
  XpkOut = Xpk.';
  WpkOut = Wpk.';
else
  XpkOut = Xpk;
  WpkOut = Wpk;  
end


%--------------------------------------------------------------------------
function y = getIZERO
% Return zero of the indexing type: double 0 in MATLAB,
% coder.internal.indexInt(0) for code generation targets.
    y = 0;
