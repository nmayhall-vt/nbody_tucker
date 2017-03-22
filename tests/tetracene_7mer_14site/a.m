#!/usr/bin/octave -qf
	

# This code takes as input, a file containing a list of all the J12 coupling elements
#	and the number of sites,  Units are given in Wavenumbers
#	i.e.,
#4
#-6237.477641
#-1.433618
#-20.721028
#-0.330428
#-6.595823
#-5863.618915
#

au2ev = 27.21165;
au2cm = 219474.63;

convert = 1;			# 1 for wavenumbers
convert = au2ev/au2cm 		# convert from wavenumbers to eV



arg_list = argv ();
if rows(arg_list) == 0
	printf(" No file specified\n" );
	exit
end
filename = arg_list{1};


##
##	Print Eigenvectors?	
print_vecs = 0


##
##	Turn S_blocks on to solve each m_s block separately
S_blocks = 1


sx = .5* [0  1; 1  0]
sy = .5* [0 -i; i  0]
sz = .5* [1  0; 0 -1]
s2 = .75*[1  0; 0  1]
s1 = sx + sy + sz
I = eye(2)

sp = sx + i*sy 		# s_+
sm = sx - i*sy 		# s_-


#
#	Read J12 values
j12 = csvread (filename);
N = j12(1,1)
J12 = zeros(N,N);
ii = 2;
for i=1:N
	for j=i+1:N
		J12(i,j) = j12(ii,1);
		J12(j,i) = j12(ii,1);
		ii += 1;
	end
end
J12 = J12 * convert

save "j12.m" J12 
exit

H = zeros(2^N);
S2 = zeros(2^N);
Sz = zeros(2^N);

printf("\n");
printf(" Size of Hamiltonian: ")
disp(size(H))
printf("\n");

for i=1:N
	I1 = 1;
	I2 = 1;
	if i>1
		I1 = eye(2^(i-1));
	endif
		
	#get diagonal of S2
	if N-i > 0
		I2 = eye(2^(N-i));
	end
	S2 = S2 + kron(I1,s2,I2);
	Sz = Sz + kron(I1,sz,I2);

	for j=i+1:N
		printf(" H(%4i %4i)\n", i,j);
		I2 = 1;
		I3 = 1;
		if j-i>1
			I2 = eye(2^(j-i-1));
		endif
		if N-j>0
			I3 = eye(2^(N-j));
		endif

		hij = kron(I1,sp,I2,sm,I3)+kron(I1,sm,I2,sp,I3)+2*kron(I1,sz,I2,sz,I3);
		H = H - J12(i,j) * hij;
		S2 = S2 + hij;
	end
end

#todo use this to block diagonalize H and solve for one M_s block at a time
Szd = diag(Sz);
[uselessVariable,permutation]=sort(Szd);
#permutation 	= flipud(permutation);
Szd=Szd(permutation);
H=H(permutation,permutation);
S2=S2(permutation,permutation);


if S_blocks == 1
	printf("\n Solving for each m_s block separately\n\n");
else
	printf("\n Solving for all m_s blocks simultaneously\n\n");
end

if S_blocks
	for S = 1:N+1
		ms = .5*N-S+1; # do this sub-block
		if ms < 0
			continue
		end
		printf("\n =========================================================\n")
		printf(" Do M_s block = %12.8f \n", ms)
		printf(" =========================================================\n")
		ms_basis = [];
		for i = 1:rows(Szd)
			if abs(Szd(i) - ms) < 1e-8
				ms_basis = [ms_basis;i];
			end
		end
		Hms = H(ms_basis,ms_basis);
		S2ms = S2(ms_basis,ms_basis);
		printf(" Size of m_s subblock of H: ")
		disp(size(Hms));
	
		[v,l] = eig(Hms);
	
		e_gs = min(diag(l));
        
		S2v = v'*S2ms*v;
        
        
		printf(" Eigenvalues:\n");
		printf(" %5s: %18s %16s  %12s\n","State", "Energy", "Relative Energy", "<S^2>")
		for i=1:columns(l)
			printf(" %5i: %18.8f %16.8f  %12.8f\n",i,l(i,i),l(i,i)-e_gs,S2v(i,i))
		end
		printf("\n");
		if print_vecs
			printf(" Eigenvectors\n");
			disp(v)
		end
	end
else
	[v,l] = eig(H);
	
	l = l-min(diag(l));
		
	e_gs = min(diag(l));

	S2v = v'*S2*v;

	printf(" Eigenvalues:\n");
	printf(" %5s: %18s %16s  %12s\n","State", "Energy", "Relative Energy", "<S^2>")
	for i=1:columns(l)
		printf(" %4i: %16.8f  %12.8f\n",i,l(i,i),S2v(i,i))
	end
	if print_vecs
		printf(" Eigenvectors\n");
		disp(v)
	end
end


