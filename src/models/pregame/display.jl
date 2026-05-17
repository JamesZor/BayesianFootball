# src/Models/PreGame/display.jl

# ==============================================================================
# PRETTY PRINTING (Pregame Models & Components)
# ==============================================================================

# ------------------------------------------------------------------------------
# 0. Mathematical Formulations
# ------------------------------------------------------------------------------
math_formula(::AbstractModelComponent) = nothing

# Interception
math_formula(::GlobalInterception) = "μ ~ μ_prior\nλ_base = exp(μ)"
math_formula(::SeasonalInterception) = "μ[s] ~ μ_prior\nλ_base = exp(μ[season])"

# Dispersion
math_formula(::GlobalDispersion) = "log_r ~ log_r_prior\nr_h = r_a = exp(log_r)"
math_formula(::HomeAwayDispersion) = "log_r, δ_r_home ~ priors\nr_a = exp(log_r), r_h = exp(log_r + δ_r_home)"

# Home Advantage
math_formula(::GlobalHomeAdvantage) = "γ_global ~ γ_prior\nλ_home *= exp(γ_global)"
math_formula(::HierarchicalTeamHomeAdvantage) = "γ_team_raw ~ Normal(0,1)\nγ_team = γ_base + (γ_team_raw * σ_γ)\nλ_home *= exp(γ_team)"
math_formula(::HierarchicalLeagueHomeAdvantage) = "γ_league_raw ~ Normal(0,1)\nγ_league = γ_base + (γ_league_raw * σ_γ)\nλ_home *= exp(γ_league)"

# Kappa
math_formula(::GlobalKappa) = "κ_global ~ κ_prior\nλ_goals = κ_global * True_xG"
math_formula(::HierarchicalTeamKappa) = "κ_team_raw ~ Normal(0,1)\nκ_linear = κ_base + (κ_team_raw * σ_κ)\nκ_team = log1p(exp(κ_linear))\nλ_goals = κ_team * True_xG"

# Dynamics
math_formula(::TimeDecayDynamics) = "α, β ~ Normal(0, σ)\nw_match = 0.5^(Δdays / half_life)"
math_formula(::MultiScaleGRW) = "α[t] ~ Normal(α[t-1], σ)\nβ[t] ~ Normal(β[t-1], σ)"

# ------------------------------------------------------------------------------
# 1. Model Components (AbstractModelComponent)
# ------------------------------------------------------------------------------
function Base.show(io::IO, ::MIME"text/plain", config::AbstractModelComponent)
    struct_name = string(nameof(typeof(config)))
    printstyled(io, struct_name, color=:green, bold=true)
    
    formula = math_formula(config)
    if !isnothing(formula)
        printstyled(io, " [Math: ", color=:light_blue, italic=true)
        indented_formula = replace(formula, "\n" => "\n         ")
        printstyled(io, indented_formula, "]\n", color=:light_blue, italic=true)
    else
        println(io)
    end
    
    # Check if there are any fields to display
    props = propertynames(config)
    if isempty(props)
        println(io, "()")
        return
    end

    println(io, "(")
    for (i, prop) in enumerate(props)
        val = getproperty(config, prop)
        is_last = (i == length(props))
        prefix = is_last ? "  └── " : "  ├── "
        
        printstyled(io, prefix, color=:light_black)
        printstyled(io, string(prop), color=:white)
        printstyled(io, " = ", color=:light_black)
        
        # Format the value slightly
        if typeof(val) <: Distribution
            printstyled(io, string(val), color=:yellow)
        elseif typeof(val) <: Number
            printstyled(io, string(val), color=:cyan)
        else
            printstyled(io, string(val), color=:normal)
        end
        println(io)
    end
end

# Fallback for inline printing
function Base.show(io::IO, config::AbstractModelComponent)
    print(io, string(nameof(typeof(config))), "()")
end

# ------------------------------------------------------------------------------
# 2. PreGame Models (AbstractNegBinModel)
# ------------------------------------------------------------------------------
function Base.show(io::IO, ::MIME"text/plain", model::AbstractNegBinModel)
    struct_name = string(nameof(typeof(model)))
    
    # Title & Type
    printstyled(io, "Model", color=:magenta, bold=true)
    printstyled(io, " [$struct_name]", color=:yellow, bold=true)
    printstyled(io, "\n (Bayesian Football Engine)\n", color=:light_black)
    println(io, "=========")

    props = propertynames(model)
    
    for (i, prop) in enumerate(props)
        val = getproperty(model, prop)
        is_last = (i == length(props))
        
        # Section Header
        printstyled(io, "[$prop]\n", color=:cyan)
        
        # For Model Components, use their display format, but indent it
        if typeof(val) <: AbstractModelComponent
            # Manually inline the component printing to handle indentation
            comp_name = string(nameof(typeof(val)))
            printstyled(io, "  $comp_name", color=:green, bold=true)
            
            formula = math_formula(val)
            if !isnothing(formula)
                printstyled(io, " [Math: ", color=:light_blue, italic=true)
                indented_formula = replace(formula, "\n" => "\n         ")
                printstyled(io, indented_formula, "]\n", color=:light_blue, italic=true)
            else
                println(io)
            end
            
            comp_props = propertynames(val)
            if isempty(comp_props)
                println(io, "()")
            else
                println(io, "(")
                for (j, c_prop) in enumerate(comp_props)
                    c_val = getproperty(val, c_prop)
                    c_is_last = (j == length(comp_props))
                    prefix = c_is_last ? "    └── " : "    ├── "
                    
                    printstyled(io, prefix, color=:light_black)
                    printstyled(io, string(c_prop), color=:white)
                    printstyled(io, " = ", color=:light_black)
                    
                    if typeof(c_val) <: Distribution
                        printstyled(io, string(c_val), color=:yellow)
                    elseif typeof(c_val) <: Number
                        printstyled(io, string(c_val), color=:cyan)
                    else
                        printstyled(io, string(c_val), color=:normal)
                    end
                    println(io)
                end
            end
        else
            # Standard Fields (Distributions, floats, etc.)
            printstyled(io, "  └── ", color=:light_black)
            if typeof(val) <: Distribution
                printstyled(io, string(val), color=:yellow)
            elseif typeof(val) <: Number
                printstyled(io, string(val), color=:cyan)
            else
                printstyled(io, string(val), color=:normal)
            end
            println(io)
        end
        
        if !is_last
            println(io)
        end
    end
end

# Compact inline show
function Base.show(io::IO, model::AbstractNegBinModel)
    struct_name = string(nameof(typeof(model)))
    print(io, struct_name, "(...)")
end