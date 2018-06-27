function [a,diff] = linearize(perts)
n = size(perts,2);
[input, counter]  = linearizeRecursive(n, 0, [], zeros(n,2^n));
output = zeros(1,counter);
diff = zeros(1,counter);
for i = 1:counter
    inputVec = input(:,i);
    output(i) = exp(inputVec'*perts');
end
[M,i_M] = max(output);
[m,i_m] = min(output);
Y = zeros(1,n);
Z = zeros(counter - 2,n);
Y(1) = (m - M)/(input(1,i_m) - input(1,i_M));
for i = 2:n
    Y(i) = (input(i,i_m) - input(i,i_M))/(input(1,i_m) - input(1,i_M));
end
ejer = 0;
for i = 1:counter
    if i ~= i_m && i ~= i_M
        Z(i - ejer,1) = output(i) - M  - Y(1)*(input(1,i) - input(1,i_M));
    else
        ejer = ejer + 1;
    end
end
for j = 2:n
    ejer = 0;
    for i = 1:counter
        if i ~= i_m && i ~= i_M
            Z(i - ejer,j) = input(j,i) - input(j,i_M)  - Y(j)*(input(1,i) - input(1,i_M));
        else
            ejer = ejer + 1;
        end
    end
end
A = Z(:,2:end);
b = Z(:,1);
B = A'*A;
a = B\(A'*b);
a_1 = Y(1) - a'*Y(2:end)';
a = [a_1;a];
a_0 = M - a'*input(:,i_M);
a = [a_0;a];
for i = 1:counter
    inputVec = input(:,i);
    diff(i) = a'*[1;inputVec] - output(i);
end