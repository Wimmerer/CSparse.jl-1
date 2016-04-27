using Metis, CSparse, Compat
using Base.Test

include("testmat.jl")

function specialize(A::SparseMatrixCSC)
    issymmetric(A) && return Symmetric(triu!(A),:U)
    ishermitian(A) && return Hermitian(triu!(A),:U)
    istril(A) && return LowerTriangular(A)
    istriu(A) && return UpperTriangular(A)
    A
end

cd("Matrix") do
    global mats = [specialize(testmat(nm)) for nm in readdir()]
end

@test size(mats[1]) == (219,85)

@test size(mats[2]) == (48,48)
const o48 = ones(48)
@test_approx_eq mats[2]*(mats[2]\o48) o48
@test_approx_eq Ac_mul_B(mats[2],Ac_ldiv_B(mats[2],o48)) o48

const bc01t = mats[2]'
@test istriu(bc01t)
@test isa(bc01t,UpperTriangular)
@test size(bc01t) == (48,48)
@test_approx_eq bc01t*(bc01t\o48) o48
@test_approx_eq Ac_mul_B(bc01t,Ac_ldiv_B(bc01t,o48)) o48

@test size(mats[3]) == (4884,4884)
const o4884 = ones(4884)
@test_approx_eq mats[3]*(mats[3]\o4884) o4884
@test_approx_eq Ac_mul_B(mats[3],Ac_ldiv_B(mats[3],o4884)) o4884

const bc16t = mats[3]'
@test istriu(bc16t)
@test isa(bc16t,UpperTriangular)
@test size(bc16t) == (4884,4884)
@test_approx_eq bc16t*(bc16t\o4884) o4884
@test_approx_eq Ac_mul_B(bc16t,Ac_ldiv_B(bc16t,o4884)) o4884

const sc1 = Symmetric(triu(mats[2].data'mats[2].data),:U)
const perm1,iperm1 = nodeND(sc1)
const tr1 = etree(sc1)
const sc2 = symperm(sc1,iperm1)
const tr2,postperm = etree(sc2,true)
const sc3 = symperm(sc2,invperm(postperm))
const tr3 = etree(sc3)

# elimination tree
## upper triangle of the pattern test matrix from Figure 4.2 of
## "Direct Methods for Sparse Linear Systems" by Tim Davis, SIAM, 2006
rowval = Int32[1,2,2,3,4,5,1,4,6,1,7,2,5,8,6,9,3,4,6,8,10,3,5,7,8,10,11]
colval = Int32[1,2,3,3,4,5,6,6,6,7,7,8,8,8,9,9,10,10,10,10,10,11,11,11,11,11,11]
A = sparse(rowval, colval, ones(length(rowval)))
p = etree(A)
P,post = etree(A, true)
@test P == p
@test P == Int32[6,3,8,6,8,7,9,10,10,11,0]
@test post == Int32[2,3,5,8,1,4,6,7,9,10,11]
@test isperm(post)

# csc_permute
A = sprand(10,10,0.2)
p = randperm(10)
q = randperm(10)
@test CSparse.csc_permute(A, invperm(p), q) == full(A)[p, q]