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
generating synthetic data
=#
tspan = (0.,2/5)
u0 = 8.
R = 14
C = 1.5
n_points = 30

p = [R/60,C]

t = range(tspan[1],tspan[2],length=n_points)

prob = ODEProblem(wk2,u0,tspan,p)
ode_data = Array(solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, saveat=t))

#ode_data += rand(n_points)/10

# simple NN as an example
dudt = Chain(
             Dense(1,50,tanh),
             Dense(50,1))

n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t,reltol=1e-7,abstol=1e-9)
ps = Flux.params(n_ode)

pred = n_ode([u0])
scatter(t,ode_data,label="data")
scatter!(t,pred[1,:],label="prediction")

loss_n_ode() = sum(abs2, ode_data .- n_ode([u0]))

data = Iterators.repeated((), 1000)
opt = ADAM(1e-2)

loss_tr = []

cb = function () #callback function to observe training
  loss = loss_n_ode()
  display(loss)
  push!(loss_tr, loss)
  # plot current prediction against data
  #cur_pred = n_ode([u0])
  #pl = scatter(t,ode_data,label="data")
  #scatter!(pl,t,cur_pred[1,:],label="prediction")
  #display(plot(pl))

end

# Display the ODE with the initial parameter values.
cb()

Flux.train!(loss_n_ode, ps, data, opt, cb = cb)

pl = scatter(t,ode_data,label="data")
scatter!(pl,t,n_ode([u0])[1,:],label="prediction")
display(plot(pl))
