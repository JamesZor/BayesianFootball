"""
Defines the concrete component types (the "tags") that serve as the
building blocks for models.
"""
module PreGameComponents

using ..PreGameInterfaces

export PoissonGoal, NegativeBinomial, AR1, RandomWalk

# --- Component Types ---

# Goal Distributions
struct PoissonGoal <: GoalDistribution end
struct NegativeBinomial <: GoalDistribution end

# Time Dynamics
struct AR1 <: TimeDynamic end
struct RandomWalk <: TimeDynamic end

end

