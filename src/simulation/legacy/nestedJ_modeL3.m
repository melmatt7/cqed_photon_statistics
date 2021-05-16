% Author: Sebastian Gitt
clear
%close all
%% Initialization
set(groot,'defaultfigureposition',[400 250 900 700])
tic
showfig1=1;

tnum=2;
t_end=20;

%plotting parameters
s=0.2;
lim1=-70;
lim2=70;
wnum=2000;

minval=6.66;
time=0;

n=1;
N=n;

go=.32;
go=0.949;
k=43;   %  3.3 - Q~100,000 , 33 - Q~10,000
k1=k/2;
k2=k/2;
g=.12; %uev
Gc=4.3;
wc=0;
we=0;
J=0;

go=1;
k=50;
g=0.1;

%jvec=J*ones(1,n*(n-1)/2);
%jvec=(normrnd(.5,.1,1,n*(n-1)/2));


Go=go*ones(1,N);
G=g*ones(1,N);
K=k*ones(1,N);
%We=normrnd(0,g*100,1,N);
%We=normrnd(0,g*100,1,N);
%We=[-2 2];

u=1.96*3.33e-30;     %2 Debye into SI units
lambda=2.9e-6;    %2.9 um wavelength
c=3e8;            %speed of light
w=2*pi*c/lambda;  %get angular frequency
hb=1.054e-34;     %hbar
eo=8.854e-12;     %permittivity of free space
er=11.7;          %relative permittivity of silicon
refractive_index=sqrt(er);       %get refractive index of Si
%u=u/3;    %our factor of 3 difference

w=linspace(lim1,lim2,wnum);

n=[1 10 100];

We_vec=normrnd(0,g*117,1,n(end));
X=normrnd(0,10e-9,1,n(end));
Y=normrnd(0,10e-9,1,n(end));
Z=normrnd(0,10e-9,1,n(end));
[d,costheta,LWoverlap]=calcD(X,Y,Z,We_vec,g);

%jvec=LWoverlap'.*3/4.*g.*lambda^3./(2*pi)^3./d.^3.*(1-3*costheta.^2);
%jvec=LWoverlap'.*u^2/(4*pi*eo)./d.^3.*(1-3*costheta.^2)*6.2e18*1e6; %joule to uev
J_vec1=0*u^2/(4*pi*eo)./d.^3.*(1-3*costheta.^2)*6.2e18*1e6; %joule to uev
J_vec2=LWoverlap'.*u^2/(4*pi*eo)./d.^3.*(1-3*costheta.^2)*6.2e18*1e6; %joule to uev
J_vec3=u^2/(4*pi*eo)./d.^3.*(1-3*costheta.^2)*6.2e18*1e6; %joule to uev

J_vec=[J_vec1,J_vec2,J_vec3];

%% Start of Loop
for jcase = 2
for index1 = 1:length(n);
N=n(index1);


G=g*ones(1,N);
We=We_vec(1:N);
jvec1=J_vec(1:(N-1)*N/2,jcase);
% jvec2=J_vec2(1:(N-1)*N/2);
% jvec3=J_vec3(1:(N-1)*N/2);


nck0=nchoosek(N,0);
nck1=nchoosek(N,1);

gnd=1;
a1=[1,zeros(1,N)];


%% First Excitation Subspace
Heff1=zeros(N+1);
%cavity loss
Heff1(1,1)=wc-1i*k/2; %+1iw comes from the diagonal contribution later
if N>0
    %atomic coupling
    Heff1(1,2:N+1)=-1i*Go;
    Heff1(2:N+1,1)=1i*Go;
end
%spontaneous emission
if N>0
    Heff1(2:N+1,2:N+1)=Heff1(2:N+1,2:N+1)+(diag(We)-1i*diag(G)/2);
end

%Dipole-dipole coupling
%jvec=J*ones(1,nck2);
%jlen=nck2;
%B=tril(ones(nck1),-1);
%B(B==1)=jvec;
%J=(B+transpose(B));
J1=genJ(jvec1);
% J2=genJ(jvec2);
% J3=genJ(jvec3);

Heff1(2:nck0+nck1,2:nck0+nck1)=Heff1(2:nck0+nck1,2:nck0+nck1)+J1;
% Heff2(2:nck0+nck1,2:nck0+nck1)=Heff2(2:nck0+nck1,2:nck0+nck1)+J2;
% Heff3(2:nck0+nck1,2:nck0+nck1)=Heff3(2:nck0+nck1,2:nck0+nck1)+J3;



%%unsorted phi1
[phi1,lambda1]=eig(Heff1);
lambda1m=lambda1;
lambda1=diag(lambda1);

norm=sqrt((sum(phi1.^2)));
phi1n=[];
for i=1:size(phi1,2)
    phi_temp=phi1(:,i)/norm(i);
    phi1n=[phi1n,phi_temp];
end
phi1v=inv(phi1);
phi1*lambda1m*phi1v;
%phi1n=phi1;


%% Rest

toc
tic
winc=(lim2-lim1)/wnum;
% for i =1:wnum
%     w=lim1+i*winc;
%     w=0;
%     Dnum=lambda1m-diag(w*ones(1,N+1));
%     D=diag(diag(Dnum).^(-1));   
%     
%     pp1=gnd'*a1'*phi1;
%     pp1v=phi1v*g;
%     t2(i)=1i*gnd'*a1*phi1*D*phi1v*a1'*gnd;
%     W(i)=w;
% end

%figure(1)
%plot(W,k1*k2*t2.*conj(t2));

% gw=0;
% gw2=0;
% Gamma=0;
% 
%% Frequency dependence g1
if showfig1==1
    t_inc=t_end/tnum;
    for i =1:wnum
        w=lim1+i*winc;
        %w=2.9;
        %w=0;
        D1num=lambda1m-diag(w*ones(1,N+1));
        D1=diag(diag(D1num).^(-1));   
        %Transmission
        t(i)=1i*gnd'*a1*phi1n*D1*phi1n.'*a1'*gnd; 
        t(i)=1i*gnd'*a1*phi1*D1*phi1v*a1'*gnd;
        W(i)=w;
    end
end
toc


%% Plotting

lim=20*s;
pts=1e4;

T=k1*k2*t.*conj(t);
%T2=k1*k2*t2.*conj(t2);
r=t-1;

tk=sqrt(k1*k2)*t;
T_2port=(tk+1).*conj(tk+1);


% if showfig1==1
%     figure(1)
%     hold off
%     %plot(W-wc,T2);
%     ylim([0 1.2]);
%     figure(1)
%     plot(W-wc,T)
%     hold on
%     %plot(W-wc,T_2port,'red')
%     xlabel('Frequency')
%     ylabel('Transmission Amplitude')
%     lgd=legend('Transmission','Reflection');
%     lgd.FontSize = 18;
% end
   

%plot(W-wc,T.*conj(T))
%hold on
%plot(W-wc,T_2port.*conj(T_2port))
set(gca,'FontSize',24)
set(gcf,'position',[10,10,750,1500])
%legend(sprintf('%d atoms', N));
plotNum=length(n);
hold on
subplot(plotNum,1,(index1))
plot(W-wc,T.*conj(T),'linewidth',2)
legend(sprintf('%d atoms', N));
lgd.FontSize = 24;
xlabel('(\omega-\omega_c)','FontSize',30)
set(gca,'FontSize',24)
set(gcf,'position',[10,10,750,1500])

end



end

%print -dpsc nestedJplots.ps







