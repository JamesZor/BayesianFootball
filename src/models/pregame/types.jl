# src/Models/PreGame/types.jl

# ==========================================
# 1. ABSTRACT COMPONENT INTERFACES
# ==========================================
abstract type AbstractModelComponent end

abstract type AbstractDispersionConfig <: AbstractModelComponent end
abstract type AbstractHomeAdvantageConfig <: AbstractModelComponent end
abstract type AbstractKappaConfig <: AbstractModelComponent end
abstract type AbstractDynamicsConfig <: AbstractModelComponent end
abstract type AbstractInterceptionConfig <: AbstractModelComponent end

# ==========================================
# 2. MASTER ARCHITECTURE TYPES
# ==========================================
# Engine model config structs have been moved to their respective engine files 
# in src/Models/PreGame/engines/ to keep them self-contained.
