include("wk.jl")
using DiffEqFlux
using Flux
using DifferentialEquations
using Random
using CSV
using DataFrames
using Optim
using FFTW
using IterTools: ncycle

#=
Learn waveforms from synthetic data, where we know an exact solution
exists
=#

#=
load real data
=#
function load_art(sample = true,
  loc = "/Users/daphne/Documents/mit/ThingsWithWK/samples.csv",
  start = 4170,
  delta = 280,
  fs = 360)

  out = CSV.read("/Users/daphne/Documents/mit/ThingsWithWK/samples.csv", DataFrame)
  col = out[!,"'ART'"][2:end]
  col_f = parse.(Float64, col)

  t = range(0.,delta/fs,length=delta)
  pressure_data =col_f[start:start+delta]

  if sample
    # list of ids at which to take pressure samples
    draw_list = randperm(length(pressure_data)-1)[1:n_points]
    draw_list = sort(draw_list)

    t = t[draw_list]
    pressure_data = pressure_data[draw_list]
  end

  t, pressure_data

end

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

NOTE do not remove parameters from the list - models implemented
in wk.jl are designed to use the same parameter list.

=#
R = 20
C = 1.2
L = 0.002

R_a = 0.77

p = [R_a/60,C, R/60, L]

n_points = 30

t_p, pressure_data = load_art()

# get diastole (decreasing part of curve)
wh = t_p .> t_p[end]*2/5
t_p_dia = t_p[wh]
pressure_dia = pressure_data[wh]

wh = t_p .<= t_p[end]*2/5
t_p_sys = t_p[wh]
pressure_sys = pressure_data[wh]

u0 = pressure_sys[1]
tspan = (minimum(t_p_sys),maximum(t_p_sys))


t_synth = range(tspan[1],tspan[2],length=length(pressure_sys))
which_model = wk2



#=================================#

# solve ODE to get synthetic data
prob = ODEProblem(which_model,u0,tspan,p)
ode_data = Array(solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, saveat=t_synth))

ode_data_dct = abs.(FFTW.rfft(ode_data,1)) #, FFTW.REDFT10))

# simple NN
dudt = FastChain((x, p) -> x,
                  FastDense(1, 32, elu),
                  FastDense(32, 16, elu),
                  FastDense(16, 8, elu),
                  FastDense(8, 1))

# initialize ODE to train for the synthetic data
n_ode = NeuralODE(dudt, tspan, saveat=t_synth)

n_ode_real = NeuralODE(dudt, tspan, Tsit5(), saveat=t_p_sys, reltol=1e-9, abstol=1e-9)



function predict_neuralode(p)
  reshape(Array(n_ode([u0], p)),:)
end

function predict_neuralode_real(p)
  reshape(Array(n_ode_real([u0], p)),:)
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, (ode_data[1:length(pred)] .- pred[:]))
    return loss, pred
end

function loss_neuralode_real(p)
    pred = predict_neuralode_real(p)
    loss = sum(abs2, (pressure_sys[1:length(pred)] .- pred[:]))
    return loss, pred
end

function dct_loss_neuralode(p)
    pred = predict_neuralode(p)
    pred_dct = abs.(FFTW.rfft(pred[:],1))
    loss = sum(abs2, (ode_data_dct - pred_dct))/length(pred)
end

loss = []

iter = 0
callback = function (p, l, pred; doplot = true)
  global iter
  iter += 1

  display(l)
  push!(loss,l)
  if doplot
    # plot current prediction against data
    plt = scatter(t_synth, ode_data, label = "data")
    scatter!(plt, t_synth, pred[:], label = "prediction")
    xlabel!(plt," time (s)")
    ylabel!(plt, "pressure (mmHg)")
    display(plot(plt))
  end

  return false
end

iter = 0
callback_real = function (p, l, pred; doplot = true)
  global iter
  iter += 1

  display(l)
  push!(loss,l)
  if doplot
    # plot current prediction against data
    plt = scatter(t_p_sys, pressure_sys, label = "data")
    scatter!(plt, t_p_sys, pred[:], label = "prediction")
    xlabel!(plt," time (s)")
    ylabel!(plt, "pressure (mmHg)")
    display(plot(plt))
  end

  return false
end


k = 10
train_loader = Flux.Data.DataLoader((ode_data, t_synth); batchsize = k)
numEpochs = 1000


result_neuralode = DiffEqFlux.sciml_train(loss_neuralode,
                                          n_ode.p,
                                          #ADAM(0.05),
                                          Optim.KrylovTrustRegion(),
                                          #_data = ncycle(train_loader, numEpochs),
                                          cb = callback,
                                          maxiters = numEpochs,
                                          allow_f_increases = true)

final_neuralode = DiffEqFlux.sciml_train(loss_neuralode_real,
                                          result_neuralode.minimizer,
                                          #ADAM(0.05),
                                          Optim.KrylovTrustRegion(),
                                          #_data = ncycle(train_loader, numEpochs),
                                          cb = callback_real,
                                          maxiters = 200,
                                          allow_f_increases = true)
