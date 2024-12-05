function white_box_jsr(A,d) 
    s = discreteswitchedsystem(A) 
    optimizer_constructor = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
    soslyapb(s, d, optimizer_constructor=optimizer_constructor, tol=1e-4, verbose=1) 
    seq = sosbuildsequence(s, d, p_0=:Primal) 
    psw = findsmp(seq) 
    return psw.growthrate 
end


function path_generate(numMode,horizon)
    pathrun = [] 
    for i in 1:numMode 
        push!(pathrun,[i]) 
    end
    if horizon > 1
        for t in 1:horizon-1
            pathrunt = []
            for p in pathrun
                for i in 1:numMode
                    push!(pathrunt,vcat(p,i))
                end
            end
            pathrun = copy(pathrunt)
        end
    end
    return pathrun
end


function path_dependent_obsv(A,C,horizon) 
    numMode = size(A,1)
    n = size(A[1],1)
    p = size(C[1],1)

    obsv = 0
    obsvsq = []

    pathlist = path_generate(numMode,horizon)
    for path in pathlist
        obsvmat = zeros(p*horizon,n)
        Arun = Matrix(I,n,n)
        for t in 1:horizon
            obsvmat[p*(t-1)+1:t*p,:] = C[path[t]]*Arun
            Arun = A[path[t]]*Arun
        end
        if rank(obsvmat) == n
            obsv = 1 
            obsvsq = path
            break
        end
    end

    return obsv,obsvsq 
end

function pathwise_obsv(A,C,horizon) 
    numMode = size(A,1)
    n = size(A[1],1)
    p = size(C[1],1)

    obsv = 1
    obsvsq = []

    pathlist = path_generate(numMode,horizon)
    for path in pathlist
        obsvmat = zeros(p*horizon,n)
        Arun = Matrix(I,n,n)
        for t in 1:horizon
            obsvmat[p*(t-1)+1:t*p,:] = C[path[t]]*Arun
            Arun = A[path[t]]*Arun
        end
        if rank(obsvmat) < n
            obsv = 0 
            obsvsq = path
            break
        end
    end
    return obsv,obsvsq 
end

