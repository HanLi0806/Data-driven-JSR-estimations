using LinearAlgebra
using DynamicPolynomials
using SwitchOnSafety
using Combinatorics
using SparseArrays
using JuMP, MosekTools
using SpecialFunctions
using ControlSystems
using Plots
using LaTeXStrings
using Convex





include("../src/RandomTrajectories.jl")
include("../src/JSRCompute.jl")
include("../src/ExplicitUpperBounds.jl")
include("../src/WhiteBoxAnalysis.jl")


dim = 3; numMode = 2; dimOut = 2; horizon = 3;noise_bound_w=0.01;noise_bound_v=0.01;

numScen_budget = 200


    #A = [2 * rand(dim, dim) .- 1 for i in 1:numMode]
    #C = [2 * rand(dimOut, dim) .- 1 for i in 1:numMode]


A= [[-0.6439  -0.4901  -0.3086; -0.6682   0.4493   0.3190; -0.3140   0.4095   0.3440],[-0.5866   0.5199  -0.2464;0.5097   0.2914   0.3825;0.2710   0.5910   0.3508]]
C= [[-0.2982  0.5891  0.2737; -0.4818 -0.8210  0.1567],[-0.2982  0.5891  0.2737;-0.4818 -0.8210  0.1567]]

jsrwhite = white_box_jsr(A, 2)
println("JSR true: $jsrwhite")
#JSR true: 1.0123785374789438

gammalist_noise = []
compensation_list = []
gammalist_free = []
gammalist_sum = []

k = 2
shift = horizon-k

traj_noise = generate_trajectories_with_noise(horizon,A,C,numScen_budget,noise_bound_w, noise_bound_v)

state0_noise = traj_noise[1:dimOut*k,:]
state_noise = traj_noise[end-dimOut*k+1:end,:]

numTraj = size(state0_noise,2)


for N in 20:20:numScen_budget
    max_norm,output_norms_max = traj_norm_each_norm_max(state0_noise[:,1:N])
    expression_values_max = each_row_obsv_norm_max(output_norms_max, k, noise_bound_w, noise_bound_v)
    nonorms_max= noise_obsv_norm_max(output_norms_max, noise_bound_v,expression_values_max,k)
    spherenoise_max = sphere_noise_max(nonorms_max, noise_bound_w,noise_bound_v,k)                                                               
    ga_noise,P_noise = data_driven_lyapb_quad(state0_noise[:,1:N],state_noise[:,1:N];horizon=1,C=1e2,ub=10,lb=0,tol=1e-4,numIter=1e2,postprocess=true)
    
    norm_result=0
    compensation_term= spherenoise_max

    ga_sum =ga_noise + compensation_term
    
    push!(gammalist_sum,ga_sum)
    push!(gammalist_noise,ga_noise)
    push!(compensation_list,compensation_term)


    println("gamma list_noise: $gammalist_noise")
    println("compensation_list: $compensation_list")
    println("gamma list_sum: $gammalist_sum")
    println(repeat('*', 80))
end


