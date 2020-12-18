include("wk.jl")
using DiffEqFlux
using Flux
using DifferentialEquations
using Random
using CSV
using DataFrames

#=
Learn waveforms from synthetic data, where we know an exact solution
exists
=#

#=
SETUP
pretty much all the params you will need to change are here
tspan = range over which to generate data
  perturbed from (0,1) so that timesteps don't fall exactly on
  parts of Q(t) which are not differentiable
u0 = initial condition
R = SVR
C = aortic compliance
R_a = aortic impedence
L = aortic inductance (blood inertia)

NOTE do not remove parameters from the list - model implemented
in wk.jl are designed to use the same parameter list.

=#

tspan = (0. + 1e-7,1. + 1e-7)
u0 = 8.
R = 14
C = 1.5
n_points = 30
L = 0.002

R_a = 0.77

p = [R_a/60,C, R/60, L]

t = range(tspan[1],tspan[2],length=n_points)

which_model = wk4

#=================================#

prob = ODEProblem(which_model,u0,tspan,p)
ode_data = Array(solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, saveat=t))

#ode_data += rand(n_points)/10

# simple NN as an example
dudt = Chain(
             Dense(1,16,tanh),
             Dense(16,32,tanh),
             Dense(32,64,tanh),
             Dense(64,64,tanh),
             Dense(64,32,tanh),
             Dense(32,16,tanh),
             Dense(16,1))

n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t,reltol=1e-7,abstol=1e-9)
ps = Flux.params(n_ode)

pred = n_ode([u0])
scatter(t,ode_data,label="data")
scatter!(t,pred[1,:],label="prediction")

loss_n_ode() = sum(abs2, ode_data .- n_ode([u0]))

data = Iterators.repeated((), 500)
opt = ADAM(0.05)

loss_tr = []

cb = function () #callback function to observe training
  loss = loss_n_ode()
  display(loss)
  push!(loss_tr, loss)
  # plot current prediction against data
  cur_pred = n_ode([u0])
  pl = scatter(t,ode_data,label="data")
  scatter!(pl,t,cur_pred[1,:],label="prediction")
  display(plot(pl))

end

# Display the ODE with the initial parameter values.
cb()

Flux.train!(loss_n_ode, ps, data, opt, cb = cb)

savefig("/Users/daphne/Documents/mit/ThingsWithWK/wk4_pre.pdf")

# save model
using BSON: @save, @load
@save "wk4_pre.bson" dudt


@load "wk4_pre.bson" dudt

# load in real data
# must input ones own file path
out = CSV.read("/Users/daphne/Documents/mit/ThingsWithWK/samples.csv", DataFrame)
col = out[!,"'ART'"][2:end]
col_f = parse.(Float64, col)

#========
start and delta are determined by inspection of the waveform
not ideal but it is what it is for now
========#
start = 4170
delta = 280

fs = 360

one_cycle =col_f[start:start+delta]
t_cycle = range(0.,delta/360,length=length(one_cycle))

plot(t_cycle,one_cycle)

#=====
now, randomly select points from the curve
you may want to comment out from the initial
  definitiion of one_cycle to t_cycle below
  when you run experiments/replicates so that
  the randomly chosen aortic pressure waveform
  data will be the same across trials.
======#

draw_list = randperm(length(one_cycle)-1)[1:n_points]
draw_list = sort(draw_list)

one_cycle = one_cycle[draw_list]
t_cycle = t_cycle[draw_list]

#=======#

n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t_cycle,reltol=1e-7,abstol=1e-9)
ps = Flux.params(n_ode)

u0 = one_cycle[1]

pred = n_ode([u0])
scatter(t_cycle,one_cycle,label="data")
scatter!(t_cycle,pred[1,:],label="prediction")


loss_n_ode_post() = sum(abs2, one_cycle[2:end] .- n_ode([u0]))

data = Iterators.repeated((), 100)
opt = ADAM(0.001)

loss_tr_post = []

cb_post = function () #callback function to observe training
  loss = loss_n_ode_post()
  display(loss)
  push!(loss_tr_post, loss)
  # plot current prediction against data
  cur_pred = n_ode([u0])
  pl = scatter(t_cycle,one_cycle,label="data")
  scatter!(pl,t_cycle,cur_pred[1,:],label="prediction")
  display(plot(pl))

end

# Display the ODE with the initial parameter values.
cb_post()

Flux.train!(loss_n_ode, ps, data, opt, cb = cb_post)

savefig("/Users/daphne/Documents/mit/ThingsWithWK/wk4_post.pdf")

@save "wk4_post.bson" dudt

#=====
Generate plots extrapolating the waveform to 10 seconds
=====#
how_long = 10 #seconds
how_much = how_long * fs

multi_cycle =col_f[start:start+how_much]
t_mc = range(0.,10.,length=how_much)

pl = plot(t_mc, multi_cycle[2:end], size=(800,350), label="data", legend=:bottomright)
xlabel!("time (s)")
ylabel!("pressure (mmHg)")

# load and plot models
for model in ["wk2_post.bson","wk3_post.bson","wk4_post.bson"]
  @load model dudt

  n_ode = NeuralODE(dudt,(0.,10.),Tsit5(),reltol=1e-7,abstol=1e-9, saveat=t_mc)
  out = n_ode([multi_cycle[1]])

  plot!(pl,t_mc,out[1,:],label=model[1:3])
end

savefig("/Users/daphne/Documents/mit/ThingsWithWK/traj.pdf")

t_1 = range(0,1,length=1000)

plot()
for model in [wk2,wk3,wk4]
  u0 = 8.
  prob = ODEProblem(model,u0,(0. + 1e-7,1. + 1e-7),p)
  ode_data = Array(solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, saveat=t_1))
  plot!(t_1[2:end],ode_data)

end

xlabel!("time (s)")
ylabel!("pressure (mmHg)")

savefig("/Users/daphne/Documents/mit/ThingsWithWK/comp.pdf")
