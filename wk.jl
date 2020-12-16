using DifferentialEquations
using ForwardDiff
using Random
using Plots

#=
Default forcing function
value for Q0 based on assuming flow in one cardiac cycle is 90 cm^3
    this can also be a learned parameter
We also assume that systole is 2/5 of a cardiac cycle
=#
function Q_def(t, HR=60., Q0=424.1)
    t = t[1]

    Tc = 60/HR
    Ts = Tc *2/5

    if (t%Tc) < Ts
        Q = Q0 * sin(π * (t % Tc)/(Ts))
    else
        Q = 0
    end
    Q
end

#=
2-Element Windkessel  model
p = parameters:
    R = vascular resistance
    C = vascular compliance
Q = forcing function (flow rate in the vessel, in mL/s)
=#
function wk2(u,p,t)
    R = p[1]
    C = p[2]
    (Q_def(t) - u/R)/C
end

function wk3(u,p,t)
    R₁ = p[1]
    C = p[2]
    R₂ = p[3]

    P = u

    Qt = Q_def(t)
    dQ = x -> ForwardDiff.derivative(Q_def,x)
    dQdt = dQ(t)

    dPdt = ((1 + R₁/R₂)*Qt + C*R₁*dQdt - P/R₂)/C
    dPdt

end

function wk4(u,p,t) #,Q=Q_def)
    R₁,C,R₂,L = p
    P = u

    Qt = Q_def(t)
    dQ = x -> ForwardDiff.derivative(Q_def,x)
    dQdt = dQ(t)
    d2Qdt2 = ForwardDiff.derivative(x -> ForwardDiff.derivative(Q_def, x), t)

    dPdt = ((1 + R₁/R₂)*Qt + (R₁*C + L/R₂)*dQdt + L*C*d2Qdt2 - P/R₂)/C
    dPdt
end
