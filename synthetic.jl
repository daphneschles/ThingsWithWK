include("wk.jl")
include("models.jl")
using DiffEqFlux
using Flux
using DifferentialEquations
using Random

#=
Learn waveforms from synthetic data, where we know an exact solution
exists
=#

#=
generate WK2 data for aortic pressure waveforms
=#
function gen_wk(tspan, u0, datasize=30, model=wk2)

    R = 14
    C = 1.5
    p = [R/60,C]

    t_save = range(tspan[1],tspan[2],length=datasize)

    prob = ODEProblem(model,u0,tspan,p)
    ode_data = Array(solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, saveat=t_save))


#=
    # get the sample points
    P_known = zeros(Float64,size(t_known))
    for i in 1:length(P_known)
        loc = findall(x->x == t_known[i],sol.t)[1]
        P_known[i] = sol.u[loc]
    end

    start = findall(x->x > 5,sol.t)[1]
    t = sol.t[start:end]
    u = sol.u[start:end]

    start = findall(x->x > 5,t_known)[1]
    t_dat = t_known[start:end]
    u_dat = P_known[start:end]

    t, u, t_dat, u_dat =#
    t_save, ode_data
end

#=======================================================#

tspan = (0.,.3)
u0 = 30.

f(u,p,t) = u

t, ode_data = gen_wk(tspan, u0, 30, f)

# simple NN as an example
dudt = Chain(x -> x.^3,
             Dense(1,50,tanh),
             Dense(50,100,tanh),
             Dense(100,1))
#=
dudt = Chain(x -> x.^3,
             Dense(1,16,tanh),
             Dense(16,32,tanh),
             Dense(32,64,tanh),
             Dense(64,32,tanh),
             Dense(32,1),) =#
n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t,reltol=1e-7,abstol=1e-9)
ps = Flux.params(n_ode)

function predict_n_ode()
  n_ode([u0])
end

pred = n_ode([u0])
scatter(t,ode_data,label="data")
scatter!(t,pred[1,:],label="prediction")

loss_n_ode() = sum(abs2, t .- predict_n_ode())

epochs = 1000
data = Iterators.repeated((), epochs)
opt = ADAM(0.1)

loss_tr = []

cb = function () #callback function to observe training
  loss = loss_n_ode()
  display(loss)
  push!(loss_tr, loss)
  # plot current prediction against data
  cur_pred = predict_n_ode()
  pl = scatter(t,ode_data,label="data")
  scatter!(pl,t,cur_pred[1,:],label="prediction")
  display(plot(pl))
end

# Display the ODE with the initial parameter values.
cb()

Flux.train!(loss_n_ode, ps, data, opt, cb = cb)
