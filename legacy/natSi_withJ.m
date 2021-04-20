T=xlsread('ey.xlsx');
ey=T(536:800,1:266)';

maxey=max(max(ey));
eyn=abs(ey/maxey);

x_um=T(1:265,1)*2901/1550;
y_um=T(268:533,1)*2901/1550;

zsim=500e-9;
xsim=(max(x_um)-min(x_um))*1e-6;
ysim=(max(y_um)-min(y_um))*1e-6;
Vsim=xsim*ysim*zsim;


%rho=1.56*1e13;

genx=xsim;
geny=ysim;
genz=zsim;

aNumv=[2 10 100];
for ccount= 1:2
for count = 1:length(aNumv);


Vcm=1e6*genx*geny*genz;

aNum=aNumv(count);
X=rand(1,aNum)*genx;
Y=rand(1,aNum)*geny;
Z=rand(1,aNum)*genz;



rho=aNum/Vcm;

rhov=aNumv/Vcm;

% s=surf(ey)
% s.EdgeColor = 'none';
% view(2)
% colorbar

%close all


%% Initialization
set(groot,'defaultfigureposition',[400 250 900 700])
showfig1=1;

tnum=2;
t_end=20;
fac=1;

%plotting parameters
s=0.2;
lim1=-100;
lim2=100;
wnum=2000;

N=aNum;



go=.949;
go=go*sqrt(fac);
k=43;
%k=go^2/g;
%k=3.377;
k1=k/2;
k2=k/2;
Gc=.000;
g=0.12;
wc=0;
we=6.87;
we=0;
J=0;
Go=zeros(1,aNumv(count));
for ai=1:aNum
    Go(ai)=go*eyn(randi(266),randi(265));
end

%jvec=J*ones(1,n*(n-1)/2);
%jvec=(normrnd(.5,.1,1,n*(n-1)/2));

C=go^2/((k+Gc)*g);
%Go=go*ones(1,N);
G=g*ones(1,N);
K=k*ones(1,N);


We=normrnd(0,0.12/2.355,1,N);
We=normrnd(0,0,1,N);


%We=normrnd(0,0.12/2.355,1,N);

%We=[we];

u=1.96*3.33e-30;     %2 Debye into SI units
lambda=2.9e-6;    %2.9 um wavelength
c=3e8;            %speed of light
w=2*pi*c/lambda;  %get angular frequency
hb=1.054e-34;     %hbar
eo=8.854e-12;     %permittivity of free space
er=11.7;          %relative permittivity of silicon
refractive_index=sqrt(er);       %get refractive index of Si
u=u/3;    %our factor of 3 difference
We_vec=normrnd(0,g*117,1,aNumv(end));

w=linspace(lim1,lim2,wnum);


nck0=nchoosek(N,0);
nck1=nchoosek(N,1);

gnd=1;
a1=[1,zeros(1,N)];


%% First Excitation Subspace
Heff1=zeros(N+1);
%cavity loss
Heff1(1,1)=wc-1i*k/2-1i*Gc/2; %+1iw comes from the diagonal contribution later
if N>0
    %atomic coupling
    Heff1(1,2:N+1)=-1i*Go;
    Heff1(2:N+1,1)=1i*Go;
end
%spontaneous emission
if N>0
    Heff1(2:N+1,2:N+1)=Heff1(2:N+1,2:N+1)+(diag(We)-1i*diag(G)/2);
end

if ccount==2
[d,costheta,LWoverlap]=calcD(X,Y,Z,We_vec,g);

jvec=LWoverlap'.*u^2/(4*pi*eo)./d.^3.*(1-3*costheta.^2)*6.2e18*1e6 %joule to uev

%Dipole-dipole coupling
%jvec=J*ones(1,nck2);
%jlen=nck2;
%B=tril(ones(nck1),-1);
%B(B==1)=jvec;
%J=(B+transpose(B));
J=genJ(jvec);
Heff1(2:nck0+nck1,2:nck0+nck1)=Heff1(2:nck0+nck1,2:nck0+nck1)+J;

end

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



%% Plotting

lim=20*s;
pts=1e4;

T=k1*k2*t.*conj(t);
%T2=k1*k2*t2.*conj(t2);

r=t-1;

tk=sqrt(k1*k2)*t;
T_2port=(tk+1).*conj(tk+1);

   
color2=get(gca,'ColorOrder');
% 
% figure(1)
% plot(W-wc,T+(1*(count-1)),'color',color2(1,:))
% hold on
% %plot(W-wc,T_2port+(1*(count-1)),'color',color2(2,:))
% ylabel('Transmission [a.u.]','FontSize',24)
% 
% xlabel('(\omega-\omega_c)/\kappa','FontSize',24)
% set(gca,'FontSize',18)
% set(gca,'ytick',[])

plotNum=length(aNumv);

 

hold on
subplot(plotNum,2,2*count-1+(ccount-1))
plot(W-wc,T)
%lgd=legend('\rho = 1.56 \cdot 10^{12} atoms', N);
%lgd=legend(sprintf('\\rho = 1.56 \\cdot 10^{%0.1d} atoms/cm^3', 12+i ));
lgd=legend(sprintf('\\rho = {%0.1d} atoms/cm^3', rho ));

lgd.FontSize = 18;
% if i==round(length(n)/2)
%     ylabel('Transmission','FontSize',24)
% end
% if i==length(n)
%     xlabel('(\omega-\omega_c)/\kappa','FontSize',24)
% 
% end
set(gca,'FontSize',18)

end

end
