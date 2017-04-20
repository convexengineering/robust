import GPModels as GM
from gpkit import VarKey

testOriginal = GM.testModel()

testSubInPlace = GM.testModel()
print('the original key is:')
print(testSubInPlace['a'].key.descr)
copy_a = VarKey(**testSubInPlace['a'].key.descr)
copy_a.key.descr['value'] = 3
copy_a.key.descr['something'] = 1  
testSubInPlace.subinplace({testSubInPlace['a'].key:copy_a})
print('the substituted key is:')
print(testSubInPlace['a'].key.descr)

solutionOriginal = testOriginal.solve(verbosity=0)
solutionSubInPlace = testSubInPlace.solve(verbosity=0)

print('the original cost:')
print(solutionOriginal.get('cost'))
print('the new cost:')
print(solutionSubInPlace.get('cost'))

print('original variables')
print(solutionOriginal.get('variables'))
print('new variables')
print(solutionSubInPlace.get('variables'))

print('the same problem was solved although I did a substitution and checked that it was successfully done!!!')