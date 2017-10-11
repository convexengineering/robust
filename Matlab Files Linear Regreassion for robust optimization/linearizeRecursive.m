function [ input, counter ] = linearizeRecursive(n, counter, inputVec, input)
if length(inputVec) == n
    counter = counter + 1;
    input(:, counter) = inputVec;
else
    for i = -1:2:1
        inputVec = [inputVec;i];
        [input, counter] = linearizeRecursive(n, counter, inputVec, input);
        inputVec = inputVec(1:end-1);
    end
end
end
