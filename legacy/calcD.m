function [d,costheta,LWoverlap] = calcD(x,y,z,We,g)
N=length(x);

d=zeros(1,N*(N-1)/2);
costheta=zeros(1,N*(N-1)/2);

d0=zeros(1,N*(N-1)/2);

Linewidth_d=zeros(1,N*(N-1)/2);
LWoverlap=zeros(1,N*(N-1)/2);


[j,i]=find(tril(ones(N),-1));  %i is row, j is column
ind=[i,j]';
%dxy(:)=sqrt( (x(ind(1,:)) -x(ind(2,:)) ).^2 +(y(ind(1,:)) -y(ind(2,:)) ).^2 );
d(:)=sqrt( (x(ind(1,:)) -x(ind(2,:)) ).^2 +(y(ind(1,:)) -y(ind(2,:)) ).^2 +(z(ind(1,:)) -z(ind(2,:)) ).^2);
d=d';


Linewidth_d(:)=We(ind(1,:))-We(ind(2,:));
LWoverlap= 4*g^2./(Linewidth_d.^2+4*g^2);


costheta(:)= (y(ind(1,:)) -y(ind(2,:)) )./d';


costheta=costheta';
end
