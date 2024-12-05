function matrix_2_norm(A::AbstractMatrix)
    return norm(A, 2)
end

function traj_norm_each_norm_max(state0)
    max_norm = 0
    max_index = 0
    for i in 1:size(state0, 2)
        current_norm = norm(state0[:, i], 2)
        if current_norm > max_norm
            max_norm = current_norm
            max_index = i
        end
    end    
    max_trajectory = state0[:, max_index]
    output_norms_max = [norm(max_trajectory[j],2) for j in eachindex(max_trajectory)]
    return max_norm, output_norms_max
end

function traj_norm_each_norm_min(state0)
    min_norm = Inf 
    min_index = 0
    for i in 1:size(state0, 2)  
        current_norm = norm(state0[:, i], 2)
        if current_norm < min_norm
            min_norm = current_norm
            min_index = i
        end
    end   
    min_trajectory = state0[:, min_index]    
    output_norms_min = [norm(min_trajectory[j],2) for j in eachindex(min_trajectory)]
    return min_norm, output_norms_min
end

function each_row_obsv_norm_max(output_norms_max, k, noise_bound_w, noise_bound_v)
    expression_values_max = Float64[]
    expression_maxvalue = output_norms_max[1] + noise_bound_v
    push!(expression_values_max, expression_maxvalue)
    for j=2:k
        expression_maxvalue = output_norms_max[j] + sum(output_norms_max[j-i+1] * noise_bound_w * (noise_bound_w + 1)^(i-2) for i in 2:j) + noise_bound_v * (noise_bound_w + 1)^(j-1)
        push!(expression_values_max, expression_maxvalue)
    end
    return expression_values_max
end

function each_row_obsv_norm_min(output_norms_min, k, noise_bound_w, noise_bound_v)
    expression_values_min = Float64[]
    expression_minvalue = output_norms_min[1] + noise_bound_v
    push!(expression_values_min, expression_minvalue)
    for j=2:k
        expression_minvalue = output_norms_min[j] + sum(output_norms_min[j-i+1] * noise_bound_w * (noise_bound_w + 1)^(i-2) for i in 2:j) + noise_bound_v * (noise_bound_w + 1)^(j-1)
        push!(expression_values_min, expression_minvalue)
    end
    return expression_values_min
end

function noise_obsv_norm_max(output_norms_max, noise_bound_v,expression_values_max,k)
    nonorms_max = Float64[]
    nonorms_max = [output_norms_max[1] + noise_bound_v]
    for i=2:(k-1)
        nonorms_max_i = sqrt(sum(expression_values_max[j]^2 for j in 1:i))
        push!(nonorms_max, nonorms_max_i)
    end
    return nonorms_max
end

function noise_obsv_norm_min(output_norms_min, noise_bound_v,expression_values_min,k)
    nonorms_min = Float64[]
    nonorms_min = [output_norms_min[1] + noise_bound_v]
    for i=2:(k-1)
        nonorms_min_i = sqrt(sum(expression_values_min[j]^2 for j in 1:i))
        push!(nonorms_min, nonorms_min_i)
    end
    return nonorms_min
end

function sphere_noise_max(nonorms_max,noise_bound_w,noise_bound_v,k)
    spherenoise_max =0
    for i in eachindex(nonorms_max)
        spherenoise_max = spherenoise_max + nonorms_max[i] * noise_bound_w
    end
        spherenoise_max = spherenoise_max + sqrt(k) * noise_bound_v 
    return spherenoise_max
end

function sphere_noise_min(nonorms_min,noise_bound_w,noise_bound_v,k)
    spherenoise_min =0
    for i in eachindex(nonorms_min)
        spherenoise_min = spherenoise_min + nonorms_min[i] * noise_bound_w
    end
        spherenoise_min = spherenoise_min + sqrt(k) * noise_bound_v
    return spherenoise_min
end


function data_driven_jsr_sensitivityanalysis_quad_tuning_explicit(gamma,P,dimin,numTraj,k,numMode,horizon,min_norm,spherenoise_min,max_norm,spherenoise_max)
    dim = size(P)[1]
    kappaP = eigmax(P)/eigmin(P)
    dv = Int64((dim+1)*dim/2)
    epsilion=beta_inc_inv(10,numTraj-9,0.99)[1]
    ep = numMode^horizon*epsilion/2
    ep2 = numMode^k*0.2/2
    ep3 = numMode^k*0.2/2
    delt = sqrt(1-beta_inc_inv((dimin-1)/2.0,0.5,2*ep2)[1])
    delt1 = sqrt(1-beta_inc_inv((dimin-1)/2.0,0.5,2*ep3)[1])
    Delt = sqrt(2-2*delt1)
    
    ck_square= (delt*(min_norm - spherenoise_min)/(max_norm + spherenoise_max)-Delt)^(-2)

    EP=1-beta_inc_inv((dimin-1)/2.0,0.5,2*sqrt((ck_square*kappaP)^(dimin-1))*ep)[1]
    jsr_bound = gamma/((sqrt(EP))^(1/(horizon-k)))
    
    return jsr_bound
end

