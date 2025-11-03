"""
Defines the concrete component types (the "tags") that serve as the
building blocks for models.
"""
module PreGameComponents

# using ..PreGameInterfaces

export PoissonGoal, NegativeBinomialGoal, AR1, RandomWalk, Static

# --- Component Types ---

# Goal Distributions
struct PoissonGoal <: GoalDistribution end
struct NegativeBinomialGoal <: GoalDistribution end

# Time Dynamics
struct AR1 <: TimeDynamic end
struct RandomWalk <: TimeDynamic end
struct Static <: TimeDynamic end 

end

