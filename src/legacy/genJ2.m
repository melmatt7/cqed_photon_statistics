% Author: Sebastian Gitt
function [J2] = genJ2(jvec)

nck2=length(jvec);
nck1=round(sqrt(2*nck2+0.25)+0.5);
N=nck1;

if N<2
    J2=[];
    return
elseif N==2
    J2=0;
    return
end

jtri=tril(ones(nck1-1));
jtri(jtri==1)=jvec;
J2=[];
for i=1:N-2
    j=jtri(i:end,i:end);

    %j0=j(1,1);
    j1=j(1:end,1);
    j2=j(2:end,2:end);

    jt=[];
    for k=1:size(j2,1)
    jt_temp=[j1(k+1:end),j1(k)*eye(size(j1,1)-k)];
    
    %jt=catpad(1,jt,jt_temp);
    jt=flip((catpad(1,flip(jt,2),flip(jt_temp,2))),2);
    end
    jt=catpad(1,j2,jt);
    
    %J2=catpad(2,J2,jt);
    J2=flip((catpad(2,flip(J2,1),flip(jt,1))),1);
    J2(isnan(J2))=0;
end
% Jtesting = J2;

% take lower triangular and make into full matrix
J2=catpad(1,0,J2);
J2=catpad(2,J2,0);
J2(isnan(J2))=0;
J2=(J2+transpose(J2));
end