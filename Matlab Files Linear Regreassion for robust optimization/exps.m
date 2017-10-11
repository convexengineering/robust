f = @(x_1,x_2,x_3,x_4,x_5)exp(-2*x_1 + x_2 + 3*x_3 - 2.3*x_4 + 0.5*x_5);
input = zeros(5,2^5);
output = zeros(1,2^5);
diff = zeros(1,2^5);
counter = 0;
for i = -1:2:1
    for j = -1:2:1
        for k = -1:2:1
            for l = -1:2:1
                for c = -1:2:1
                    counter = counter + 1;
                    output(counter) = f(i,j,k,l,c);
                    input(:,counter) = [i;j;k;l;c];
                end
            end
        end
    end
end
[M,i_M] = max(output);
[m,i_m] = min(output);
A = [1, input(:,i_M)'];
A = [A;[1, input(:,i_m)']];
A = [A;[1, input(:,1)']];
A = [A;[1, input(:,2)']];
A = [A;[1, input(:,3)']];
A = [A;[1, input(:,6)']];

b = [M;m;output(1);output(2);output(3);output(6)];

a = A\b;

p = @(x_1,x_2,x_3,x_4,x_5) a(1) + a(2)*x_1 + a(3)*x_2 + a(4)*x_3 + a(5)*x_4 + a(6)*x_5;
counter = 0;

for i = -1:2:1
    for j = -1:2:1
        for k = -1:2:1
            for l = -1:2:1
                for c = -1:2:1
                    counter = counter + 1;
                    ejer = p(i,j,k,l,c) - f(i,j,k,l,c);
                    diff(counter) = ejer;
                end
            end
        end
    end
end