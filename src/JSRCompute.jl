function data_driven_lyapb_quad(state0,state;horizon=1,C=1e1,ub=10,lb=0,tol=1e-4,numIter=1e2,postprocess=false)
  numTraj = size(state0)[2]
  dim = size(state0)[1]
iter = 1
gammaU = ub
gammaL = lb
while gammaU-gammaL > tol && iter < numIter
    iter += 1
    gamma = (gammaU + gammaL)/2
    solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
    model = Model(solver)
    @variable(model, P[1:dim, 1:dim] in PSDCone())
    @constraint(model, P >= Matrix(I,dim,dim),PSDCone())
    @objective(model, Min, 0)
    for i in 1:numTraj
      @constraint(model, state[:,i]'*P*state[:,i] <= gamma^(2*horizon)*state0[:,i]'*P*state0[:,i])
    end
    @constraint(model, P <= C*Matrix(I,dim,dim),PSDCone())
    JuMP.optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
      gammaU=gamma
    else
      gammaL=gamma
    end
end
gamma = gammaU

if postprocess == true 
    solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
    model = Model(solver)
    @variable(model, P[1:dim, 1:dim] in PSDCone())
    @constraint(model, P >= Matrix(I,dim,dim),PSDCone())

    @objective(model, Min, sum(P[:].^2)) 
    for i in 1:numTraj
      @constraint(model, state[:,i]'*P*state[:,i] <= gamma^(2*horizon)*state0[:,i]'*P*state0[:,i])
    end
    @constraint(model, P <= C*Matrix(I,dim,dim),PSDCone()) 
    JuMP.optimize!(model)
    return gamma, value.(P)
else
    return gamma
end

end
